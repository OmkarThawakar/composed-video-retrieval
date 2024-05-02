import ast
from pathlib import Path

import pandas as pd
import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import transform_test, transform_train
from src.data.utils import FrameLoader, id2int, pre_caption
from src.tools.files import write_txt
from src.tools.utils import print_dist

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning


class WebVidCoVRDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        annotation: dict = {"train": "", "val": ""},
        vid_dirs: dict = {"train": "", "val": ""},
        emb_dirs: dict = {"train": "", "val": ""},
        mm_emb_dirs: dict = {"train": "", "val": ""},
        image_size: int = 384,
        emb_pool: str = "query",
        iterate: str = "pth2",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.emb_pool = emb_pool
        self.iterate = iterate
        self.vid_query_method = vid_query_method
        self.vid_frames = vid_frames

        self.transform_train = transform_train(image_size)
        self.transform_test = transform_test(image_size)

        self.data_train = WebVidCoVRDataset(
            transform=self.transform_train,
            annotation=annotation["train"],
            vid_dir=vid_dirs["train"],
            emb_dir=emb_dirs["train"],
            mm_emb_dir=mm_emb_dirs["train"],
            split="train",
            emb_pool=self.emb_pool,
            iterate=self.iterate,
            vid_query_method=self.vid_query_method,
            vid_frames=self.vid_frames,
        )
        self.data_val = WebVidCoVRDataset(
            transform=self.transform_test,
            annotation=annotation["val"],
            vid_dir=vid_dirs["val"],
            emb_dir=emb_dirs["val"],
            mm_emb_dir=mm_emb_dirs["val"],
            split="val",
            emb_pool=self.emb_pool,
            iterate=self.iterate,
            vid_query_method=self.vid_query_method,
            vid_frames=self.vid_frames,
        )

    def prepare_data(self):
        # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
        # download data, pre-process, split, save to disk, etc...
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class WebVidCoVRTestDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        annotation: str,
        vid_dirs: str,
        emb_dirs: str,
        mm_emb_dirs:str,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 384,
        emb_pool: str = "query",
        iterate: str = "pth2",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.emb_pool = emb_pool
        self.iterate = iterate
        self.vid_query_method = vid_query_method
        self.vid_frames = vid_frames

        self.transform_test = transform_test(image_size)

        self.data_test = WebVidCoVRDataset(
            transform=self.transform_test,
            annotation=annotation,
            vid_dir=vid_dirs,
            emb_dir=emb_dirs,
            mm_emb_dir=mm_emb_dirs,
            split="test",
            emb_pool=self.emb_pool,
            iterate=self.iterate,
            vid_query_method=self.vid_query_method,
            vid_frames=self.vid_frames,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class WebVidCoVRDataset(Dataset):
    def __init__(
        self,
        transform,
        annotation: str,
        vid_dir: str,
        emb_dir: str,
        mm_emb_dir:str,
        split: str,
        max_words: int = 30,
        emb_pool: str = "query",
        iterate: str = "pth2",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
    ) -> None:
        super().__init__()

        self.transform = transform

        self.annotation_pth = annotation
        assert Path(annotation).exists(), f"Annotation file {annotation} does not exist"
        self.df = pd.read_csv(annotation)

        self.vid_dir = Path(vid_dir)
        self.emb_dir = Path(emb_dir)
        self.mm_emb_dir = Path(mm_emb_dir)
        assert self.vid_dir.exists(), f"Image directory {self.vid_dir} does not exist"
        assert self.emb_dir.exists(), f"Embedding directory {emb_dir} does not exist"
        assert self.mm_emb_dir.exists(), f"Multimodel Embedding directory {emb_dir} does not exist"

        assert split in [
            "train",
            "val",
            "test",
        ], f"Invalid split: {split}, must be one of train, val, or test"
        self.split = split

        vid_pths = self.vid_dir.glob("*/*.mp4")
        emb_pths = self.emb_dir.glob("*/*.pth")
        mm_emb_pths = self.mm_emb_dir.glob("*/*.pth")

        id2vidpth = {
            vid_pth.parent.stem + "/" + vid_pth.stem: vid_pth for vid_pth in vid_pths
        }
        id2embpth = {
            emb_pth.parent.stem + "/" + emb_pth.stem: emb_pth for emb_pth in emb_pths
        }
        id2mmembpth = {
            mm_emb_pth.parent.stem + "/" + mm_emb_pth.stem: mm_emb_pth for mm_emb_pth in mm_emb_pths
        }

        assert len(id2vidpth) > 0, f"No videos found in {vid_dir}"
        assert len(id2embpth) > 0, f"No embeddings found in {emb_dir}"
        assert len(id2mmembpth) > 0, f"No embeddings found in {emb_dir}"

        self.df["path1"] = self.df["pth1"].apply(lambda x: id2vidpth.get(x, None))  # type: ignore
        self.df["path2"] = self.df["pth2"].apply(lambda x: id2embpth.get(x, None))  # type: ignore
        self.df["mmembpath"] = self.df["pth2"].apply(lambda x: id2mmembpth.get(x, None))  # type: ignore

        # Count unique missing paths
        missing_pth1 = self.df[self.df["path1"].isna()]["pth1"].unique().tolist()
        missing_pth1.sort()
        total_pth1 = self.df["pth1"].nunique()

        missing_pth2 = self.df[self.df["path2"].isna()]["pth2"].unique().tolist()
        missing_pth2.sort()
        total_pth2 = self.df["pth2"].nunique()

        if len(missing_pth1) > 0:
            print_dist(
                f"Missing {len(missing_pth1)} pth1's ({len(missing_pth1)/total_pth1 * 100:.1f}%), saving them to missing_pth1-{split}.txt"
            )
            if split == "test":
                raise ValueError(
                    f"Missing {len(missing_pth1)} pth1's ({len(missing_pth1)/total_pth1 * 100:.1f}%) in test split"
                )
            write_txt(missing_pth1, f"missing_pth1-{split}.txt")
        if len(missing_pth2) > 0:
            print_dist(
                f"Missing {len(missing_pth2)} pth2's ({len(missing_pth2)/total_pth2 * 100:.1f}%), saving them to missing_pth2-{split}.txt"
            )
            if split == "test":
                raise ValueError(
                    f"Missing {len(missing_pth2)} pth2's ({len(missing_pth2)/total_pth2 * 100:.1f}%) in test split"
                )
            write_txt(missing_pth2, f"missing_pth2-{split}.txt")

        # Remove missing paths
        self.df = self.df[self.df["path1"].notna()]
        self.df = self.df[self.df["path2"].notna()]
        self.df = self.df[self.df["mmembpath"].notna()]
        self.df.reset_index(drop=True, inplace=True)

        self.max_words = max_words

        assert emb_pool in [
            "middle",
            "mean",
            "query",
        ], f"Invalid emb_pool: {emb_pool}, must be one of middle, mean, or query"
        self.emb_pool = emb_pool

        if iterate in ["idx", "triplets"]:
            iterate = "idx"
            self.df["idx"] = self.df.index
        self.iterate = iterate
        self.target_txts = self.df[iterate].unique()
        assert iterate in self.df.columns, f"{iterate} not in {Path(annotation).stem}"
        self.df.sort_values(iterate, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df["int1"] = self.df["pth1"].apply(lambda x: id2int(x, sub="0"))
        self.df["int2"] = self.df["pth2"].apply(lambda x: id2int(x, sub="0"))
        self.pairid2ref = self.df["int1"].to_dict()
        assert (
            self.df["int1"].nunique() == self.df["pth1"].nunique()
        ), "int1 is not unique"
        assert (
            self.df["int2"].nunique() == self.df["pth2"].nunique()
        ), "int2 is not unique"
        # int2id is a dict with key: int1, value: pth1
        self.int2id = self.df.groupby("int1")["pth1"].apply(set).to_dict()
        self.int2id = {k: list(v)[0] for k, v in self.int2id.items()}

        self.pairid2tar = self.df["int2"].to_dict()
        self.df.set_index(iterate, inplace=True)
        self.df[iterate] = self.df.index

        if split == "test":
            assert (
                len(self.target_txts) == self.df.shape[0]
            ), "Test split should have one caption per row"

        assert vid_query_method in [
            "middle",
            "random",
            "sample",
        ], f"Invalid vid_query_method: {vid_query_method}, must be one of middle, random, or sample"
        self.frame_loader = FrameLoader(
            transform=self.transform, method=vid_query_method, frames_video=vid_frames
        )

    def __len__(self) -> int:
        return len(self.target_txts)

    def __getitem__(self, index):
        target_txt = self.target_txts[index]
        ann = self.df.loc[target_txt]
        if ann.ndim > 1:
            ann = ann.sample()
            ann = ann.iloc[0]

        reference_pth = str(ann["path1"])
        reference_vid = self.frame_loader(reference_pth)

        caption = pre_caption(ann["edit"], self.max_words)

        target_pth = str(ann["path2"])
        target_emb = torch.load(target_pth).cpu()

        target_pth_mmemb = str(ann["mmembpath"])
        target_mmemb = torch.load(target_pth_mmemb).cpu()

        description = str(ann["caption"])
        webvid_caption = str(ann["txt1"])

        if self.emb_pool == "middle":
            target_emb = target_emb[len(target_emb) // 2]
        elif self.emb_pool == "mean":
            target_emb = target_emb.mean(0)
        elif self.emb_pool == "query":
            vid_scores = ast.literal_eval(str(ann["scores"]))
            if len(vid_scores) == 0:
                vid_scores = [1.0] * len(target_emb)
            vid_scores = torch.Tensor(vid_scores)
            vid_scores = (vid_scores / 0.1).softmax(dim=0)
            target_emb = torch.einsum("f,fe->e", vid_scores, target_emb)
            target_mmemb = torch.einsum("f,fe->e", vid_scores, target_mmemb)

        return reference_vid, target_emb, caption, index, target_mmemb, description, webvid_caption
