from models import create_model
from rc_utils.model_utils import print_model_param_names

model = create_model()
print_model_param_names(model)