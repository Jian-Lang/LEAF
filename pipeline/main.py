import os
import sys
from datetime import datetime
from pathlib import Path

import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import wandb
from pipeline.evaluate import detection_evaluate
from pipeline.finetune import finetune
from pipeline.utils.core_utils import calculate_md5


class Runner:
    def __init__(self, cfg: DictConfig, config_md5: str):
        self.strategy = cfg.strategy
        self.dataset = cfg.dataset
        self.model = cfg.model
        self.short_name = cfg.short_name
        self.mark = cfg.mark
        self.peft_cfg = cfg.peft_cfg
        self.trainer_cfg = cfg.trainer_cfg
        self.reason_trainer_cfg = cfg.reason_trainer_cfg
        self.dataset_cfg = cfg.dataset_cfg
        self.num_img = cfg.num_img
        self.eval_cfg = OmegaConf.select(cfg, "eval_cfg", default={})
        self.config_md5 = config_md5
        self.save_path = f"models/{self.dataset}-{self.strategy}-{self.short_name}-{self.mark}-{config_md5}"
        self.result_path = Path(
            f"result/distill-detection/{self.dataset}-{self.strategy}-{self.short_name}-{self.mark}-{config_md5}.jsonl"
        )
        self.eval_strategy = cfg.eval_strategy

    def run(self):
        match self.strategy:
            case "reason":
                finetune(
                    dataset_name=self.dataset,
                    stage="reason",
                    model_name=self.model,
                    load_path=None,
                    save_path=self.save_path,
                    dataset_cfg=self.dataset_cfg,
                    peft_cfg=self.peft_cfg,
                    trainer_cfg=self.reason_trainer_cfg,
                )
            case "explain":
                finetune(
                    dataset_name=self.dataset,
                    stage="explain",
                    model_name=self.model,
                    load_path=None,
                    save_path=self.save_path,
                    dataset_cfg=self.dataset_cfg,
                    peft_cfg=self.peft_cfg,
                    trainer_cfg=self.trainer_cfg,
                )
            case "reason-explain":
                tmp_save_path = (
                    f"models/{self.dataset}-reason-{self.short_name}-{self.mark}-{self.config_md5}"
                )
                finetune(
                    dataset_name=self.dataset,
                    stage="reason",
                    model_name=self.model,
                    load_path=None,
                    save_path=tmp_save_path,
                    dataset_cfg=self.dataset_cfg,
                    peft_cfg=self.peft_cfg,
                    trainer_cfg=self.reason_trainer_cfg,
                )
                finetune(
                    dataset_name=self.dataset,
                    stage="explain",
                    model_name=self.model,
                    load_path=tmp_save_path,
                    save_path=self.save_path,
                    dataset_cfg=self.dataset_cfg,
                    peft_cfg=self.peft_cfg,
                    trainer_cfg=self.trainer_cfg,
                )
            case _:
                raise ValueError(f"Invalid strategy: {self.strategy}")

        match self.eval_strategy:
            case "detection":
                result = detection_evaluate(
                    dataset_name=self.dataset,
                    result_path=self.result_path,
                    load_path=self.save_path,
                    num_img=self.num_img,
                    **self.eval_cfg,
                )
                wandb.log({"acc": result["acc"], "m-f1": result["f1"]})
            case _:
                raise ValueError(f"Invalid eval strategy: {self.eval_strategy}")


@hydra.main(version_base=None, config_path="cfg", config_name="MHClipEN_explain")
def main(cfg: DictConfig):
    config_str = OmegaConf.to_yaml(cfg)
    config_md5 = calculate_md5(config_str)[:4]
    os.environ["WANDB_PROJECT"] = "LEAF"
    os.environ["WANDB_LOG_MODEL"] = "false"
    wandb.init(project="LEAF", config=OmegaConf.to_container(cfg, resolve=True))

    log_path = (
        Path(f"log/{cfg.short_name}-{cfg.dataset}-{cfg.mark}-{config_md5}")
        / f"{datetime.now().strftime('%m%d-%H%M%S')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(log_path, level="DEBUG")
    logger.add(sys.stdout, level="INFO")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Log Path: {log_path}")

    runner = Runner(cfg, config_md5)
    runner.run()


if __name__ == "__main__":
    main()
