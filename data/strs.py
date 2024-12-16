class TaskStrs:
    LESION_DETECTION = 'lesion-detection'
    NEGBIO_CLASSIFICATION = "negbio-classification"
    CHEXPERT_CLASSIFICATION = "chexpert-classification"
    FIXATION_GENERATION = "fixation-generation"
    
    AGE_REGRESSION = "age-regression"
    TEMPERATURE_REGRESSION = "temperature-regression"
    HEARTRATE_REGRESSION = "heartrate-regression"
    RESPRATE_REGRESSION = "resprate-regression"
    O2SAT_REGRESSION = "o2sat-regression"
    SBP_REGRESSION = "sbp-regression"
    DBP_REGRESSION = "dpb-regression"
    ACUITY_REGRESSION = "acuity-regression"
    GENDER_CLASSIFICATION = "gender-classification"
    
    XRAY_CLINICAL_CL = "xray-clinical-cl"

class SourceStrs:
    XRAYS = "xrays"
    CLINICAL = "clinical"
    FIXATIONS = "fixations"
    CLINICAL_1D = "clinical_1d"


class FusionStrs:
    ElEMENTWISE_SUM = "element-wise sum"
    HADAMARD_PRODUCT = "hadamard product"
    NO_ACTION = "no-action"
    CONCAT = "concat"
    CONCAT_DEFORM = "concat_deform"
    CONCAT_WITH_BLOCK = "concat_with_block"
    CONCAT_WITH_TOKENMIXER = "concat_with_token_mixer"
    CONCAT_WITH_BLOCK_TOKENMIXER = "concat_with_block_token_mixer"
