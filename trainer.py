from pytorch_lightning.cli import LightningCLI

if __name__ == "__main__":
    LightningCLI(
        parser_kwargs={
            "fit": {"default_config_files": ["config.fit.yaml"]},
        },
    )
