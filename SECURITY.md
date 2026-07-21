# Security policy

## Scope

This repository contains offline research utilities and a PyTorch model
definition. It does not serve HTTP requests, download datasets, execute
notebooks or bundle model checkpoints.

Only load checkpoints that you created or obtained from a trusted source.
PyTorch checkpoint files may use Python pickle and can execute code during
deserialization. The public project deliberately does not provide a generic
checkpoint loader.

## Reporting

Please report a concrete issue privately through
<https://german-leontiev.com/contact/>. Do not attach patient data, medical
images, credentials or proprietary checkpoints.

Only the latest `main` revision is supported.
