import os
import random
import warnings
import pandas as pd


from tqdm import tqdm
from src.data.utils import *

warnings.filterwarnings("ignore")
mimic_data_path = os.path.join('.','data','raw','mimic-iv-meds','data','train')
mimic_metadata_path = os.path.join('.','data','raw','mimic-iv-meds','metadata',)
labs_metadata_path = os.path.join('.','resources','mimic-mapping')
labs_dimension_path = os.path.join('.','data','raw')

icd_mapping_files_path = os.path.join('.','resources','icd-code-conversion')
medications_files_path = os.path.join('.','resources','medications')
labs_files_path = os.path.join('.','resources','labs')

interm_path = os.path.join('.','data','intermediate')
processed_path = os.path.join('.','data','processed')
final_path = os.path.join('.','data','meds_final')

# 11503051: not sorted
# 11503085: sorted
# 11503098: shuffled
files = os.listdir(mimic_data_path)
# files = sorted(os.listdir(mimic_data_path))
random.shuffle(files)
completed_files = os.listdir(interm_path)
files = [file for file in files if file not in completed_files]

for file in files:
    if file.endswith('.parquet'):


        medgemma_rankings_d = pd.read_csv(os.path.join(icd_mapping_files_path,'1-2-many_gems_d_ranked1.csv'))
        gpt_rankings_d = pd.read_csv(os.path.join(icd_mapping_files_path,'1-2-many_gems_d_ranked_gpt1.csv'))

        medgemma_rankings_p = pd.read_csv(os.path.join(icd_mapping_files_path,'1-2-many_gems_p_ranked1.csv'),dtype={"icd9_code": str, "icd10_code": str})
        gpt_rankings_p = pd.read_csv(os.path.join(icd_mapping_files_path,'1-2-many_gems_p_ranked_gpt1.csv'),dtype={"icd9_code": str, "icd10_code": str})

        gems_cm_labeled = pd.read_csv(os.path.join(icd_mapping_files_path,'gems_cm_labeled.csv'))
        high_level_d = pd.read_csv(os.path.join(icd_mapping_files_path,'high_level_d.csv'))

        gems_pcs_labeled = pd.read_csv(os.path.join(icd_mapping_files_path,'gems_pcs_labeled.csv'),dtype={"icd9": str, "icd10": str})
        high_level_p = pd.read_csv(os.path.join(icd_mapping_files_path,'high_level_p.csv'),dtype={"icd9": str, "icd10": str})

        cleaned_medications = pd.read_csv(os.path.join(medications_files_path,'cleaned_medications.csv'))

        labs_metadata = pd.read_csv(os.path.join(labs_metadata_path,'d_labitems_to_loinc.csv'))
        labs_dimension = pd.read_csv(os.path.join(labs_dimension_path,'d_labitems.csv')) 
        cleaned_lab_values = pd.read_csv(os.path.join(labs_files_path,'lab_textual_mapping.csv'))

        icu_items_dimensions = pd.read_csv(os.path.join(labs_metadata_path,'d_items.csv'))


        print(f"Processing file: {file}")
        shard = pd.read_parquet(os.path.join(mimic_data_path, file))

        # Patient without hospital admission
        patients_without_hadm = []
        for pid in tqdm(shard.subject_id.unique()):
            patient = shard[shard.subject_id == pid]
            patient_hadm_id = patient.hadm_id.unique()
            if patient_hadm_id.shape[0] == 1:
                patients_without_hadm.append(int(pid))

        # filter
        shard = shard[shard.subject_id.isin(patients_without_hadm) == False].reset_index(drop=True)    
        print('1-patients without hadm_id removed')

        # remove table name from code
        shard['table'] = shard.code.apply(lambda x: x.split('//')[-1])
        shard.code = shard.code.apply(lambda x: '//'.join(x.split('//')[:-1]))
        print('2-table names removed')


        # handle patient race

        # unify races
        shard['race'] = shard.code.apply(lambda x: x.split('//')[1] if x.startswith('RACE') else np.nan)
        shard['code'] = shard.code.apply(lambda x: clean_race(x) if x.startswith('RACE') else x)

        # Step 1: Create base sequential 'filter' column
        shard["filter"] = range(len(shard))

        # Step 2: Identify RACE rows
        race_mask = shard["code"].str.contains("RACE", na=False)

        # Step 3: Assign the same filter value for consecutive RACE rows
        filter_values = []
        group_id = -1
        for i, is_race in tqdm(enumerate(race_mask)):
            if i == 0 or not is_race or not race_mask.iloc[i - 1]:
                group_id += 1
            filter_values.append(group_id)
        shard["filter"] = filter_values

        # Step 4: Collapse duplicates by keeping first row per filter group
        shard = shard.groupby("filter", as_index=False).first()
        shard.drop(columns=['filter'],inplace=True)
        patients_with_multiple_races = []
        for pid in tqdm(shard.subject_id.unique()):
            patient = shard[shard.subject_id == pid]
            patient_races = patient[patient.code.str.startswith('RACE')]
            if patient_races.shape[0] > 2:
                patients_with_multiple_races.append(int(pid))
                
        print(len(patients_with_multiple_races))
        print('3-race cleaned')

        # clean outpatient measuerments
        cleaned_timelines = []
        for pid in tqdm(shard.subject_id.unique()):
            patient = shard[shard.subject_id == pid]
            cleaned_patient = clean_outpatient_measurements(patient)
            cleaned_timelines.append(cleaned_patient)

        shard = pd.concat(cleaned_timelines).reset_index(drop=True)
        del(cleaned_timelines)
        print('4-removed outpatient measurements')

        # clean empty admissions

        empty_hadms = []
        patients_with_empty_hadms = []

        for pid in tqdm(shard.subject_id.unique()):
            patient = shard[shard.subject_id == pid]
            admissions = patient.hadm_id.dropna().unique()

            for hid in admissions:
                admission = patient[patient.hadm_id == hid].reset_index(drop=True)
                age_rows = admission[admission.code.str.startswith('AGE_AT_ADMISSION')]

                if not age_rows.empty:
                    idx = age_rows.index[0]
                    if idx + 1 < len(admission):
                        next_code = admission.iloc[idx + 1].code
                        if next_code.startswith('DISCHARGE-FROM-HOSPITAL'):
                            empty_hadms.append(int(hid))
                            patients_with_empty_hadms.append(int(pid))

        patients_with_single_empty = []
        for pid in patients_with_empty_hadms:
            patient = shard[shard.subject_id == pid]
            hids = patient.hadm_id.unique()
        #     print(hids.shape[0],pid)
            if hids.shape[0] == 2:
                patients_with_single_empty.append(pid)
                
        patients_with_empty_hadms = [pid for pid in patients_with_empty_hadms if \
                                    pid not in patients_with_single_empty]

        shard = shard[shard.subject_id.isin(patients_with_single_empty) == False].reset_index(drop=True)

        shard = shard[(shard.hadm_id.isin(empty_hadms) == False) & 
                    (shard.out_id.isin(empty_hadms) == False)].reset_index(drop=True) 
        
        print('5-empty admissions cleaned')


        # handle procedure codes
        shard = push_procedure_and_sort(shard)
        print('6-procedures cleaned')


        # get first rankings only
        medgemma_rankings_d = medgemma_rankings_d.groupby('icd9_code').first().reset_index()
        gpt_rankings_d = gpt_rankings_d.groupby('icd9_code').first().reset_index()

        # drop rank and reason column
        medgemma_rankings_d = medgemma_rankings_d.iloc[:,:-2]
        gpt_rankings_d = gpt_rankings_d.iloc[:,:-2]

        #merge both tables
        merged_ranking_d = pd.merge(gpt_rankings_d,medgemma_rankings_d,how='outer',on='icd9_code',suffixes=('_gpt','_medgemma'))

        # apply final mapping 
        merged_ranking_d['final'] = merged_ranking_d.apply(lambda x: make_mapping_decesion(x.icd10_code_gpt,x.icd10_code_medgemma),axis=1)


        # exact mapping
        exact_d = gems_cm_labeled[gems_cm_labeled.label == 'exact']
        exact_mappings_d = dict(zip(exact_d['icd9'],exact_d['icd10']))

        # one to one mapping
        one_to_one_d = gems_cm_labeled[gems_cm_labeled.label == 'one_to_one']
        one_to_one_mappings_d = dict(zip(one_to_one_d['icd9'],one_to_one_d['icd10']))

        # one to many mapping
        one_to_many_mappings_d = dict(zip(merged_ranking_d['icd9_code'],merged_ranking_d['final']))

        # high level mapping
        high_level_mapping_d = dict(zip(high_level_d.icd9,high_level_d.icd10))

        all_mappings = exact_mappings_d | one_to_one_mappings_d | one_to_many_mappings_d | high_level_mapping_d


        # all possible mapping
        shard['icd9_to_icd10_d'] = shard.diag_icd_code.map(all_mappings)

        # combination mapping
        combinations = gems_cm_labeled[gems_cm_labeled.label == 'combination']
        combinations = combinations[combinations.scenario == 1]
        combinations.groupby(['icd9','choice_list']).first().reset_index()
        combinations = combinations[['icd9','icd10']]
        shard = shard.merge(combinations,left_on='diag_icd_code',right_on='icd9',how='left')
        shard.icd9_to_icd10_d = shard.apply(lambda x: x['icd10'] if pd.notna(x['icd10']) else x['icd9_to_icd10_d'], axis=1)

        # no map elimination
        no_map_code = gems_cm_labeled[gems_cm_labeled.label == 'no_map'].icd9.unique()
        shard = shard[shard.diag_icd_code.isin(no_map_code) == False].reset_index(drop=True)

        shard = shard[shard.diag_icd_code.isin(['V451','V854','V138','V51','V127','V109','V581','V608','V152','V251','V122',
                                                'V610','V155','V403','V135']) == False].reset_index(drop=True)

        # unify names
        shard.code = shard.apply(lambda x:'//'.join([x.code_type,x.icd9_to_icd10_d]) if x.code.startswith('DIAGNOSIS-ICD//9') else x.code,axis=1)
        shard.code = shard.apply(lambda x:'//'.join([x.code_type,x.diag_icd_code]) if x.code.startswith('DIAGNOSIS-ICD//10') else x.code,axis=1)

        shard.drop(columns=['icd9','icd10'],inplace=True)


        print('7-diagnosis codes cleaned')

        # procedure codes mapping
        # get first rankings only
        medgemma_rankings_p = medgemma_rankings_p.groupby('icd9_code').first().reset_index()
        gpt_rankings_p = gpt_rankings_p.groupby('icd9_code').first().reset_index()

        # drop rank and reason column
        medgemma_rankings_p = medgemma_rankings_p.iloc[:,:-2]
        gpt_rankings_p = gpt_rankings_p.iloc[:,:-2]

        #merge both tables
        merged_ranking_p = pd.merge(gpt_rankings_p,medgemma_rankings_p,how='outer',on='icd9_code',suffixes=('_gpt','_medgemma'))

        # apply final mapping 
        merged_ranking_p['final'] = merged_ranking_p.apply(lambda x: make_mapping_decesion(x.icd10_code_gpt,x.icd10_code_medgemma),axis=1)


        # exact mapping
        exact_p = gems_pcs_labeled[gems_pcs_labeled.label == 'exact']
        exact_mappings_p = dict(zip(exact_p['icd9'],exact_p['icd10']))

        # one to one mapping
        one_to_one_p = gems_pcs_labeled[gems_pcs_labeled.label == 'one_to_one']
        one_to_one_mappings_p = dict(zip(one_to_one_p['icd9'],one_to_one_p['icd10']))

        # one to many mapping
        one_to_many_mappings_p = dict(zip(merged_ranking_p['icd9_code'],merged_ranking_p['final']))

        # high level mapping
        high_level_mapping_p = dict(zip(high_level_p.icd9,high_level_p.icd10))

        all_mappings = exact_mappings_p | one_to_one_mappings_p | one_to_many_mappings_p | high_level_mapping_p


        # all possible mapping
        shard['icd9_to_icd10_p'] = shard.proc_icd_code.map(all_mappings)

        # combination mapping
        combinations = gems_pcs_labeled[gems_pcs_labeled.label == 'combination']
        combinations = combinations[combinations.scenario == 1]
        combinations.groupby(['icd9','choice_list']).first().reset_index()
        combinations = combinations[['icd9','icd10']]
        shard = shard.merge(combinations,left_on='proc_icd_code',right_on='icd9',how='left')
        shard.icd9_to_icd10_p = shard.apply(lambda x: x['icd10'] if pd.notna(x['icd10']) else x['icd9_to_icd10_p'], axis=1)

        # no map elimination
        no_map_code = gems_pcs_labeled[gems_pcs_labeled.label == 'no_map'].icd9.unique()
        shard = shard[shard.proc_icd_code.isin(no_map_code) == False].reset_index(drop=True)

        shard = shard[shard.proc_icd_code.isin(['857']) == False].reset_index(drop=True)

        # unify names
        shard.code = shard.apply(lambda x:'//'.join([x.code_type,x.icd9_to_icd10_p]) if x.code.startswith('PROCEDURE-ICD//9') else x.code,axis=1)
        shard.code = shard.apply(lambda x:'//'.join([x.code_type,x.proc_icd_code]) if x.code.startswith('PROCEDURE-ICD//10') else x.code,axis=1)

        shard.drop(columns=['icd9','icd10'],inplace=True)


        print('8-procedure codes cleaned')

        # Medications cleaning
        cleaned_medications = cleaned_medications.replace(np.nan,None)
        medication_mapping = dict(zip(cleaned_medications.original_name,cleaned_medications.clean))
        shard['clean_medication'] = shard.medication.map(medication_mapping)
        shard = shard[shard.clean_medication.isna() == False]
        shard.code = shard.apply(lambda x:'//'.join([x.code_type,x.clean_medication]) if x.code.startswith('MEDICATION') else x.code,axis=1)
        shard = shard[shard.code.str.startswith('MEDICATION//UNK') == False].reset_index(drop=True)

        print('7-medications cleaned')


        # Microbiology cleaning
        shard.code = shard.apply(lambda x:'//'.join([x.code_type, str(int(x.micro_test_itemid)),x.micro_test_name]) if x.code.startswith('MICROBIOLOGY') else x.code,axis=1)
        shard.micro_org_name = shard.micro_org_name.apply(lambda x: None if x == 'NEGATIVE' else x)
        shard.micro_org_name = shard.micro_org_name.apply(lambda x: None if x == 'NO GROWTH' else x)
        shard.micro_spec_type_desc = shard.micro_spec_type_desc.apply(lambda x: 'TISSUE' if x == 'XXX' else x)
        shard.micro_spec_type_desc = shard.micro_spec_type_desc.apply(lambda x: 'BLOOD CULTURE' if x == '' else x)
        shard = shard[shard.micro_org_name != 'CANCELLED'].reset_index(drop=True)
        shard.text_value = shard.apply(lambda x: 'NEGATIVE' if (x.micro_org_name == None) & (x.code_type == 'MICROBIOLOGY') else x.text_value, axis=1)

        print('8-microbiology cleaned')

        not_labs = [50807, 50812, 50829, 50845, 50886, 50887, 50888, 50897, 50919,
                    50923, 50932, 50933, 50934, 50947, 50955, 50979, 50984, 50985,
                    51038, 51056, 51103, 51107, 51129, 51571, 51591, 51599, 51600,
                    51601, 51602, 51603, 51604, 51608, 51612, 51671, 51678, 51698,
                    51699, 51700, 51702, 51703, 51706, 51712, 51717, 51718, 51719,
                    51720, 51727, 51752, 51757, 51759, 51760, 51771, 51796, 51806,
                    51827, 51828, 51830, 51831, 51839, 51901, 51905, 51906, 51907,
                    51924, 51953, 51955, 51978, 51993, 51995, 51997, 51998, 52014,
                    52016, 52023, 52025, 52033, 52036, 52043, 52066, 52067, 52068,
                    52118, 52161, 52186, 52195, 52229, 52230, 52231, 52232, 52233,
                    52234, 52235, 52236, 52237, 52238, 52239, 52240, 52241, 52242,
                    52243, 52244, 52245, 52246, 52247, 52248, 52249, 52250, 52251,
                    52252, 52253, 52254, 52287, 52288, 52289, 52290, 52313, 52314,
                    52315, 52334, 52370, 52371, 52372, 52374, 52392, 52393, 52405,
                    52406, 52412, 52415, 52418, 52419, 52420, 52421, 52422, 52423,
                    53127, 51564, 51597, 51605, 51657, 51658, 51659, 51660, 51661, 
                    51663, 51664, 51665, 51686, 51732, 51733, 51734, 51735, 51736,
                    51737, 51762, 51763, 51764, 51765, 51766, 51767, 51768, 51772,
                    51789, 51817, 51849, 51850, 51851, 51852, 51856, 51857, 51902,
                    51903, 51904, 51908, 51909, 51916, 51939, 51954, 51956, 51970,
                    51971, 51973, 52004, 52005, 52006, 52007, 52008, 52009, 52010,
                    52011, 52012, 52018, 52019, 52020, 52021, 52080, 52081, 52083,
                    52084, 52110, 52136, 52137, 52147, 52148, 52153, 52169, 52191,
                    52194, 52215, 52217, 52317, 52318, 52333, 52394, 52395, 52396,
                    52397, 52398, 52399, 52400, 52401, 52402, 52424, 52425, 52426,
                    52427, 53122, 51662, 50827, 50828, 51509,51513]
        

        # Handle labs
        labs_metadata = labs_metadata.rename(columns={'itemid (omop_source_code)':'itemid'})




        lab_label = dict(zip(labs_dimension['itemid'], labs_dimension['label']))
        lab_fluid = dict(zip(labs_dimension['itemid'], labs_dimension['fluid']))
        lab_category = dict(zip(labs_dimension['itemid'], labs_dimension['category']))
        lab_description = dict(zip(labs_metadata['itemid'], labs_metadata['omop_concept_name']))
        lab_frequency = dict(zip(labs_metadata['itemid'], labs_metadata['labevents_row_count']))
        lab_valueuom = dict(zip(labs_metadata['itemid'], labs_metadata['valueuom']))


        shard['lab_label'] =  shard.lab_itemid.map(lab_label)
        shard['lab_fluid'] =  shard.lab_itemid.map(lab_fluid)
        shard['lab_category'] =  shard.lab_itemid.map(lab_category)
        shard['lab_description'] =  shard.lab_itemid.map(lab_description)
        shard['lab_frequency'] =  shard.lab_itemid.map(lab_frequency)
        shard['lab_valueuom'] =  shard.lab_itemid.map(lab_valueuom)

        shard.lab_label = shard.lab_label.apply(lambda x: x.split(', ')[0] if type(x) == str 
                                                and ((x.split(', ')[-1] in list(labs_metadata.fluid.unique())) 
                                                or (x.split(', ')[-1] in ['Body Fluid', 'Other Fluid'])) 
                                                else x)


        shard = shard[shard.lab_itemid.isin(not_labs) == False].reset_index(drop=True)
        all_values = shard[shard.code_type == 'LAB'].text_value.unique()
        target_values = cleaned_lab_values.original_values.unique()
        values = []
        for value in all_values:
            if value not in target_values:
                values.append(value)
        values = [item for item in values if item is not None]
        shard = shard[shard.text_value.isin(values) == False].reset_index(drop=True)

        cleaned = dict(zip(cleaned_lab_values.original_values, cleaned_lab_values.clean))
        shard.text_value = shard.text_value.apply(lambda x: cleaned[x] if x in cleaned.keys() else x)

        numeric = cleaned_lab_values[cleaned_lab_values.numric.notna()]
        cleaned_numeric = dict(zip(numeric.clean,numeric.numric))
        shard.numeric_value = shard.apply(lambda x: cleaned_numeric[x.text_value] if (x.text_value in cleaned_numeric.keys()) & (x.code_type == 'LAB') else x.numeric_value,axis=1)
        shard.text_value = shard.apply(lambda x: None if (x.text_value in cleaned_numeric.keys()) & (x.code_type == 'LAB') else x.text_value, axis=1)

        shard.text_value = shard.apply(lambda x: 'UNKNOWN' if ((pd.isna(x.numeric_value) and pd.isna(x.text_value))
                                                        or
                                                            (pd.isna(x.numeric_value) and x.text_value == '___')) 
                                                        and
                                                            (x.code_type == 'LAB')
                                                        else 
                                                            x.text_value, axis=1)

        shard.code = shard.apply(lambda x: '//'.join([x.code,x.lab_fluid,x.lab_label]) if x.code.startswith('LAB//') else x.code,axis= 1)
        print('9-labs cleaned')

        # # handle ICU procedures
        shard = shard[shard.category != '7-Communication'].reset_index(drop=True)
        shard.code = shard.apply(lambda x:'//'.join([x.code_type,str(int(x.itemid)),x.abbreviation]) if x.code.startswith('ICU-PROCEDURE') else x.code,axis=1)

        print('10-ICU procedures cleaned')

        # ICU fluids_output processing
        shard.numeric_value = shard.apply(lambda x: abs(x.numeric_value) if x.code_type == 'ICU-FLUID-OUTPUT' else x.numeric_value, axis=1)
        shard.code = shard.apply(lambda x:'//'.join([x.code_type,str(int(x.itemid)),x.abbreviation]) if x.code_type == 'ICU-FLUID-OUTPUT' else x.code,axis=1)

        print('11-ICU fluids output cleaned')

        # handle infusions
        shard.code = shard.apply(lambda x:'//'.join([x.code_type,str(int(x.itemid)),x.abbreviation]) if x.code_type == 'ICU-INFUSION' else x.code,axis=1)
        shard.numeric_value = shard.apply(lambda x: x.amount if x.code_type == 'ICU-INFUSION' else x.numeric_value,axis=1)
        print('12-ICU infusions cleaned')

        # handle ICU chart 
        shard.code = shard.apply(lambda x:'//'.join([x.code_type,str(int(x.itemid)),x.abbreviation]) if x.code_type == 'ICU-CHART' else x.code,axis=1)
        print('13-ICU charts cleaned')

        # unify sequence id
        shard['seq_id'] = shard.apply(lambda x: next((x[col] for col in ['out_id', 'er_id', 'hadm_id', 'disch_id'] if pd.notna(x[col])), np.nan),axis=1)
        print('14-sequence id unified')

        # unify value column
        # shard['value'] = shard.apply(lambda x: x.numeric_value if pd.notna(x.numeric_value) else x.text_value,axis=1)
        # print('15-value column unified')


        # reorder columns
        columns = ['subject_id', 'seq_id', 'out_id', 'er_id', 'hadm_id', 'icustay_id', 'disch_id',  'time', 'code', 
                 'numeric_value', 'text_value', 'itemid', 'died_in_hosp', 'icu_los', 'admission_type',
                 'admission_location', 'discharge_location', 'diag_version', 'diag_icd_code', 'diag_seq_num', 
                 'drg_severity', 'drg_mortality',  'drg_type', 'drg_code', 'priority', 'specimen_id', 
                 'lab_lower_limit', 'lab_upper_limit', 'lab_flag', 'lab_unit', 'lab_itemid', 'gender', 
                 'route', 'frequency', 'doses_per_24_hrs', 'medication', 'proc_seq_num', 'proc_version',
                 'proc_icd_code', 'micro_specimen_id', 'micro_org_name', 'micro_test_name', 'micro_spec_type_desc', 
                 'micro_test_itemid', 'icu_care_unit',  'category', 'label', 'abbreviation', 'rate', 'unit', 
                 'amount', 'amountuom', 'ordercategorydescription', 'ordercategoryname','secondaryordercategoryname', 
                 'ordercomponenttypedescription', 'table', 'race', 'code_type', 'icd9_to_icd10_d', 'icd9_to_icd10_p',
                 'clean_medication', 'lab_label', 'lab_fluid', 'lab_category', 'lab_description', 'lab_frequency']

        shard = shard[columns]
        print('15-columns reordered')

        # handle hadm_id
        shard.hadm_id = shard.apply(lambda x: np.nan if (x.hadm_id == x.er_id) or (x.hadm_id == x.disch_id) else x.hadm_id, axis= 1)
        print('16-hadm_id handled')

        shard.to_parquet(os.path.join(interm_path, file), index=False)
        print(f"File {file} processed and saved to {interm_path}")
        
        print('+'*100)
        print('+'*100)




if len(os.listdir(processed_path)) < 365:
    # names = list(range(365))
    # names = [(str(name) + '.parquet') for name in names]
    # pairs = create_pairs_from_list(names)
    

    # num = 0
    # for pair in tqdm(pairs):
    #     if len(pair) == 2:
    for file in os.listdir(interm_path):        
            shard = pd.read_parquet(os.path.join(interm_path,file))
            # shard_2 = pd.read_parquet(os.path.join(interm_path,pair[1]))
            
            shard.out_id = shard.apply(lambda x: np.nan if x.out_id == x.disch_id else x.out_id, axis= 1)
            # shard_2.out_id = shard_2.apply(lambda x: np.nan if x.out_id == x.disch_id else x.out_id, axis= 1)
            
            # shard_1.text_value = shard_1.apply(lambda x: 'UNKNOWN' if x.numeric_value == 999999.00 else x.text_value, axis = 1)
            # shard_2.text_value = shard_1.apply(lambda x: 'UNKNOWN' if x.numeric_value == 999999.00 else x.text_value, axis = 1)
            
            # shard_1.numeric_value = shard_1.numeric_value.apply(lambda x: np.nan if x == 999999.00 else x)
            # shard_2.numeric_value = shard_2.numeric_value.apply(lambda x: np.nan if x == 999999.00 else x)
            
            # shard = pd.concat([shard_1,shard_2],ignore_index=True)
            
            shard.to_parquet(os.path.join(processed_path,f'{file}'))
        
        # elif len(pair) > 2:
            
        #     shard_1 = pd.read_parquet(os.path.join(interm_path,pair))
        #     shard_1.out_id = shard_1.apply(lambda x: np.nan if x.out_id == x.disch_id else x.out_id, axis= 1)
        #     shard_1.text_value = shard_1.apply(lambda x: 'UNKNOWN' if x.numeric_value == 999999.00 else x.text_value, axis = 1)
        #     shard_1.numeric_value = shard_1.numeric_value.apply(lambda x: np.nan if x == 999999.00 else x)
        #     shard_1.to_parquet(os.path.join(processed_path,f'{num}.parquet'))
        # num+=1



if len(os.listdir(final_path)) < 365:
    files = os.listdir(processed_path)
    random.shuffle(files)
    for file in tqdm(files):
        print(f"Finalizing file: {file}")
        shard = pd.read_parquet(os.path.join(processed_path, file))
        
        all_patients = []
        for idx in tqdm(shard.subject_id.unique()):
            patient = shard[shard.subject_id == idx]
            patient = add_emer_outp_boundries(patient)
            patient = add_time_tokens(patient)
            
            all_patients.append(patient)
            
        shard = pd.concat(all_patients,ignore_index=True)
        shard.to_parquet(os.path.join(final_path, file), index=False)