#adapted from: https://github.com/alawryaguila/multi-view-AE
from schema import Schema, And, Or, Optional, SchemaError, Regex

SUPPORTED_ENCODERS = [
            "cnn.Encoder",
            "cnn.VariationalEncoder",
            "cnn.ConditionalVariationalEncoder"
        ]

SUPPORTED_DECODERS = [
            "cnn.Decoder",
            "cnn.ConditionalDecoder",
        ]

SUPPORTED_DISCRIMINATORS = [
            "mlp.Discriminator"
        ]

SUPPORTED_DISTRIBUTIONS = [
            "distributions.Default",
            "distributions.Normal",
            "distributions.MultivariateNormal",
            "distributions.Bernoulli",
            "distributions.Laplace"
        ]


UNSUPPORTED_ENC_DIST = [
            "distributions.Bernoulli"
        ]

UNSUPPORTED_PRIOR_DIST = [
            "distributions.Default",
            "distributions.Bernoulli"
        ]

def return_or(params=[], msg="invalid"):
    assert(len(params) > 0)

    p = ', '.join('"{0}"'.format(str(w)) for w in params)
    return f"Or({p}, error='{msg}')"

def return_regexor(params=[], msg="invalid"):
    assert(len(params) > 0)
    p = '|'.join('{0}'.format(str(w)) for w in params)
    return f"Regex(r'(.*?).({p})$', error='{msg}')"

def list_sub(a, b):
    return list(set(a) - set(b))

config_schema = Schema({
    "model": {
        "save_model": bool,
        
        "seed_everything": bool,
        "seed": And(int, lambda x: 0 <= x <= 4294967295),
        "z_dim": int,
        "learning_rate": And(float, lambda x: 0 < x < 1),
        "sparse": bool,
        "threshold": Or(And(float, lambda x: 0 < x < 1), 0),
        Optional("eps"): And(float, lambda x: 0 < x <= 1e-10),
        Optional("beta"): And(Or(int, float), lambda x: x > 0),
        Optional("K"): And(int, lambda x: x >= 1),
        Optional("alpha"): And(Or(int, float), lambda x: x > 0),
        Optional("private"): bool,
    },
    "datamodule": {
        "_target_": str, 
        "batch_size": Or(And(int, lambda x: x > 0), None),
        "is_validate": bool,
        "train_size": And(float, lambda x: 0 < x < 1),
        "dataset": {"_target_": str, }
    },
    "encoder": {
        "default": {
            "_target_" : eval(return_regexor(params=SUPPORTED_ENCODERS,
                            msg="encoder._target_: unsupported or invalid encoder")),
            Optional("hidden_layer_dim"): [And(int, lambda x: x > 0)],
            Optional(Regex(r'^layer\d$')) : {
                "layer": str, 
            },
            "bias": bool,
            "non_linear": bool,
            "enc_dist": {
                    "_target_": eval(return_regexor(params=list_sub(SUPPORTED_DISTRIBUTIONS, UNSUPPORTED_ENC_DIST),
                            msg="encoder.enc_dist._target_: unsupported or invalid encoder distribution"))
            }
        },
        Optional(Regex(r'^enc\d$')) : {
            "_target_" : eval(return_regexor(params=SUPPORTED_ENCODERS,
                            msg="encoder._target_: unsupported or invalid encoder")),
            Optional("hidden_layer_dim"): [And(int, lambda x: x > 0)],
            Optional(Regex(r'^layer\d$')) : {
                "layer": str, 
            },
            "bias": bool,
            "non_linear": bool,
            "enc_dist": {
                    "_target_": eval(return_regexor(params=list_sub(SUPPORTED_DISTRIBUTIONS, UNSUPPORTED_ENC_DIST),
                            msg="encoder.enc_dist._target_: unsupported or invalid encoder distribution"))
            }
        }
    },
    "decoder": {
        "default": {
            "_target_" : eval(return_regexor(params=SUPPORTED_DECODERS,
                            msg="decoder._target_: unsupported or invalid decoder")),
            Optional("hidden_layer_dim"): [And(int, lambda x: x > 0)],
            Optional(Regex(r'^layer\d$')) : {
                "layer": str, 
            },
            "bias": bool,
            "non_linear": bool,
            Optional("init_logvar"): Or(int, float),
            "dec_dist": {
                    "_target_": eval(return_regexor(params=SUPPORTED_DISTRIBUTIONS,
                            msg="decoder.dec_dist._target_: unsupported or invalid decoder distribution"))
            }
        },
        Optional(Regex(r'^dec\d$')) : {
            "_target_" : eval(return_regexor(params=SUPPORTED_DECODERS,
                            msg="decoder._target_: unsupported or invalid decoder")),
            Optional("hidden_layer_dim"): [And(int, lambda x: x > 0)],
            Optional(Regex(r'^layer\d$')) : {
                "layer": str, 
            },
            "bias": bool,
            "non_linear": bool,
            Optional("init_logvar"): Or(int, float),
            "dec_dist": {
                    "_target_": eval(return_regexor(params=SUPPORTED_DISTRIBUTIONS,
                            msg="decoder.dec_dist._target_: unsupported or invalid decoder distribution"))
            }
        }
    },
    Optional("discriminator"): {
        "_target_" : eval(return_regexor(params=SUPPORTED_DISCRIMINATORS,
                        msg="discriminator._target_: unsupported or invalid discriminator")),
        "hidden_layer_dim": [And(int, lambda x: x > 0)],
        "bias": bool,
        "non_linear": bool,
        "dropout_threshold": Or(0, And(float, lambda x: 0 < x < 1))
    },
    "prior": {
       "_target_" : eval(return_regexor(params=list_sub(SUPPORTED_DISTRIBUTIONS, UNSUPPORTED_PRIOR_DIST),
                        msg="prior._target_: unsupported or invalid prior")),
       "loc":  Or(float, [float]),
       "scale": Or(And(float, lambda x: x > 0), [And(float, lambda x: x > 0)])
    },
    "trainer": {
       "_target_" : "pytorch_lightning.Trainer",
       "max_epochs": And(int, lambda x: x > 0),
       "deterministic": bool,
       "log_every_n_steps": And(int, lambda x: x > 0),
    },
    "callbacks": {
        "model_checkpoint": {   
           "_target_" : "pytorch_lightning.callbacks.ModelCheckpoint",
           "monitor": Or("train_loss", "val_loss"), # see training_step() and validation_step()
           "mode": Or("min", "max"),
           "save_last": bool,
           "dirpath": str   
        },
        "early_stopping": { 
           "_target_" : "pytorch_lightning.callbacks.EarlyStopping",
           "monitor": Or("train_loss", "val_loss"), # see training_step() and validation_step()
           "mode": Or("min", "max"),
           "patience": And(int, lambda x: x > 0),
           "min_delta": float,
           "verbose": bool
        }
    },
    "logger": {
       "_target_" : str,
       "save_dir": str 
    }
}, ignore_extra_keys=True)
