# encryption/encryption_utils.py
import tenseal as ts

def get_tenseal_context():
    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=4096,
        plain_modulus=1032193  # Must be prime and > max input value
    )
    context.generate_galois_keys()
    context.generate_relin_keys()
    context.global_scale = 2 ** 40
    return context
