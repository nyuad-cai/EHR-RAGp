import os
import yaml
import argparse
import chromadb

import lightning.pytorch as lt 

from transformers import RoFormerConfig, RoFormerModel  
from src.data.datasets import SequencesGenerator, limits
from src.vectordb.databases import EHREmbedder, EmbedCollator, ChromaEHREmbeddingFunction, VectorDBUploader
from src.models.utils import load_config_with_env


lt.seed_everything(24, workers=True)
parser = argparse.ArgumentParser(description='MLM pretraining command line interface')

parser.add_argument('--config-path', type=str)


args = parser.parse_args()


config = load_config_with_env(args.config_path)


seq_gen = SequencesGenerator(
            tokenizer_path=config['seq_gen']['tokenizer_path'],
            chunk_length=config['seq_length'],
            overlap=config['overlap'],
            return_numeric=False,
            return_text=False)


cfg = RoFormerConfig(vocab_size=seq_gen.tokenizer.vocab_size,
                        hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        max_position_embeddings=1536,
                        pad_token_id=seq_gen.tokenizer.pad_id,
                        type_vocab_size= 28,
                        visit_vocab_size= 102,
                        stage_vocab_size= 5)


embedder = EHREmbedder(
    config=cfg,
    backbone=RoFormerModel,
    ckpt_path=config['ckpt_path'],
    pooling=config['embedder']['pool'],                        
    normalize=True,
    use_numeric=False,
    use_time=False,)

collate_fn = EmbedCollator()

ehr_embedf = ChromaEHREmbeddingFunction(embedder=embedder, 
                                        collate_fn=collate_fn, 
                                        batch_size=None)

client = chromadb.PersistentClient(path=config['client']['path'] + config['main_window'] + '_' + str(config['seq_length']) + '_' + str(config['overlap']))

collection = client.get_or_create_collection(
            name=config['main_window'],
            embedding_function=ehr_embedf,
            metadata={"seq_length": config['seq_length'],
                      "overlap": config['overlap'],},
            configuration={"hnsw": {"space": "cosine",
                                    "ef_construction": 200}})




uploader = VectorDBUploader(
    collection=collection,
    seq_gen=seq_gen,
    main_window=config['main_window'],
    seq_length=config['seq_length'],
    limits=limits,
    data_idx_path=config['dataset']['data_idx_path'],
    data_path=config['dataset']['data_path'],
    use_time=True,
    use_numeric=True,
    add_cls_per_chunk=True)

uploader.upsert_chunks()
print(f"Finished uploading to vectordb {collection.name}.")