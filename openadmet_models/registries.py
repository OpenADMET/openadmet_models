# IMPORTANT: order matters here, make sure to import all the registered classes first
# before importing the registry classes

from openadmet_models.eval.cross_validate import *  # noqa: F401 F403
from openadmet_models.eval.eval_base import evaluators  # noqa: F401 F403

# evaluators
from openadmet_models.eval.regression import *  # noqa: F401 F403
from openadmet_models.features.chemprop import *  # noqa: F401 F403
from openadmet_models.features.combine import *  # noqa: F401 F403
from openadmet_models.features.feature_base import featurizers  # noqa: F401 F403

# featurizers
from openadmet_models.features.molfeat_fingerprint import *  # noqa: F401 F403
from openadmet_models.features.molfeat_properties import *  # noqa: F401 F403
from openadmet_models.models.chemprop.chemprop import *  # noqa: F401 F403

# models
from openadmet_models.models.gradient_boosting.lgbm import *  # noqa: F401 F403
from openadmet_models.models.model_base import models  # noqa: F401  F403
from openadmet_models.split.scaffold import *  # noqa: F401 F403
from openadmet_models.split.split_base import splitters  # noqa: F401 F403

# splitters
from openadmet_models.split.vanilla import *  # noqa: F401 F403
from openadmet_models.trainer.lightning import *  # noqa: F401 F403

# trainers
from openadmet_models.trainer.sklearn import *  # noqa: F401 F403
from openadmet_models.trainer.trainer_base import trainers  # noqa: F401 F403
from openadmet_models.util.log import logger  # noqa: F401 F403
