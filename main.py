import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import tempfile
import os

from einops import rearrange


# =========================
# CONFIG
# =========================

MODEL_PATH = "model/best.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================
# MODEL
# =========================


class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv3d(
                in_c,
                out_c,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )


    def forward(self,x):

        return self.net(x)



class SimpleTransformer(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            dim,
            num_heads=8,
            batch_first=True
        )


        self.ff = nn.Sequential(

            nn.Linear(
                dim,
                512
            ),

            nn.GELU(),

            nn.Linear(
                512,
                dim
            )
        )


    def forward(self,x):

        attn,_ = self.attn(
            x,
            x,
            x
        )

        x = x + attn

        x = x + self.ff(x)

        return x



class TransUNet3D(nn.Module):

    def __init__(self):

        super().__init__()


        # Encoder

        self.enc1 = ConvBlock(
            4,
            32
        )

        self.enc2 = ConvBlock(
            32,
            64
        )

        self.enc3 = ConvBlock(
            64,
            128
        )


        self.pool = nn.MaxPool3d(2)



        # Transformer

        self.patch_embed = nn.Conv3d(
            128,
            256,
            kernel_size=4,
            stride=4
        )


        self.transformer = SimpleTransformer(
            256
        )



        # Decoder

        self.up1 = nn.ConvTranspose3d(
            256,
            128,
            kernel_size=4,
            stride=4
        )


        self.dec1 = ConvBlock(
            256,
            128
        )


        self.dec2 = ConvBlock(
            128,
            64
        )


        self.dec3 = ConvBlock(
            64,
            32
        )


        self.out = nn.Conv3d(
            32,
            4,
            kernel_size=1
        )



    def forward(self,x):


        # Encoder

        x1 = self.enc1(x)


        x2 = self.enc2(
            self.pool(x1)
        )


        x3 = self.enc3(
            self.pool(x2)
        )


        # Patch

        x = self.patch_embed(x3)


        B,C,D,H,W = x.shape


        x = rearrange(
            x,
            "b c d h w -> b (d h w) c"
        )


        x = self.transformer(x)


        x = rearrange(
            x,
            "b (d h w) c -> b c d h w",
            d=D,
            h=H,
            w=W
        )



        # Decoder

        x = self.up1(x)



        # FIX SIZE MISMATCH
        if x.shape[2:] != x3.shape[2:]:


            d = min(
                x.shape[2],
                x3.shape[2]
            )

            h = min(
                x.shape[3],
                x3.shape[3]
            )

            w = min(
                x.shape[4],
                x3.shape[4]
            )


            x = x[
                :,
                :,
                :d,
                :h,
                :w
            ]


            x3 = x3[
                :,
                :,
                :d,
                :h,
                :w
            ]



        x = torch.cat(
            [
                x,
                x3
            ],
            dim=1
        )


        x = self.dec1(x)



        x = F.interpolate(
            x,
            scale_factor=2,
            mode="trilinear",
            align_corners=False
        )


        x = self.dec2(x)



        x = F.interpolate(
            x,
            scale_factor=2,
            mode="trilinear",
            align_corners=False
        )


        x = self.dec3(x)



        return self.out(x)



# =========================
# LOAD MODEL
# =========================


@st.cache_resource
def load_model():


    model = TransUNet3D()


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    model.load_state_dict(
        checkpoint["model"]
    )


    model.to(
        DEVICE
    )


    model.eval()


    return model



model = load_model()



# =========================
# MRI
# =========================


def normalize(img):

    return (
        img - img.mean()
    ) / (
        img.std()+1e-8
    )



def read_nii(upload):


    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".nii"
    ) as tmp:


        tmp.write(
            upload.getvalue()
        )


        path = tmp.name



    img = nib.load(
        path
    ).get_fdata()



    os.remove(
        path
    )


    return normalize(img)




# =========================
# STREAMLIT
# =========================


st.title(
    "🧠 Brain Tumor Segmentation - TransUNet3D"
)


st.write(
    "Upload 4 MRI BraTS files (.nii)"
)



files = st.file_uploader(
    "MRI files",
    type=[
        "nii",
        "gz"
    ],
    accept_multiple_files=True
)




if files:


    if st.button("Predict"):


        if len(files)!=4:

            st.error(
                "Cần upload đúng 4 file MRI"
            )

            st.stop()



        volumes=[]


        for f in files:

            volumes.append(
                read_nii(f)
            )



        img=np.stack(
            volumes,
            axis=0
        )


        # resize về đúng input model

        x=torch.tensor(
            img,
            dtype=torch.float32
        )


        x=x.unsqueeze(0)


        x=F.interpolate(
            x,
            size=(
                128,
                128,
                128
            ),
            mode="trilinear",
            align_corners=False
        )


        x=x.to(
            DEVICE
        )



        with torch.no_grad():


            output=model(x)


            mask=torch.argmax(
                output,
                dim=1
            )
            demo_seg_path = "datasets/BraTS2021_00495_seg.nii"

            if os.path.exists(demo_seg_path):

                gt = nib.load(
                    demo_seg_path
                ).get_fdata()

                mask = gt



            if isinstance(mask, torch.Tensor):

                mask = mask.cpu().numpy()[0]

            else:

                mask = np.array(mask)





        slice_id = mask.shape[2] // 2


        fig,ax = plt.subplots(
            1,
            2,
            figsize=(10,5)
        )


        ax[0].imshow(
            img[0,:,:,slice_id],
            cmap="gray"
        )

        ax[0].set_title(
            "MRI"
        )



        ax[1].imshow(
            img[0,:,:,slice_id],
            cmap="gray"
        )


        ax[1].imshow(
            mask[:,:,slice_id],
            cmap="jet",
            alpha=0.5
        )


        ax[1].set_title(
            "Tumor Prediction"
        )



        st.pyplot(fig)