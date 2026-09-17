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

    def __init__(self,in_c,out_c):
        super().__init__()

        self.net=nn.Sequential(

            nn.Conv3d(
                in_c,
                out_c,
                3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        return self.net(x)



class SimpleTransformer(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.attn=nn.MultiheadAttention(
            dim,
            8,
            batch_first=True
        )


        self.ff=nn.Sequential(

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

        a,_=self.attn(
            x,
            x,
            x
        )

        x=x+a
        x=x+self.ff(x)

        return x




class TransUNet3D(nn.Module):

    def __init__(self):

        super().__init__()


        self.enc1=ConvBlock(4,32)
        self.enc2=ConvBlock(32,64)
        self.enc3=ConvBlock(64,128)

        self.pool=nn.MaxPool3d(2)


        self.patch_embed=nn.Conv3d(
            128,
            256,
            4,
            4
        )


        self.transformer=SimpleTransformer(
            256
        )


        self.up1=nn.ConvTranspose3d(
            256,
            128,
            4,
            4
        )


        self.dec1=ConvBlock(
            256,
            128
        )


        self.dec2=ConvBlock(
            128,
            64
        )


        self.dec3=ConvBlock(
            64,
            32
        )


        self.out=nn.Conv3d(
            32,
            4,
            1
        )



    def forward(self,x):


        x1=self.enc1(x)

        x2=self.enc2(
            self.pool(x1)
        )

        x3=self.enc3(
            self.pool(x2)
        )


        x=self.patch_embed(x3)


        B,C,D,H,W=x.shape


        x=rearrange(
            x,
            "b c d h w -> b (d h w) c"
        )


        x=self.transformer(x)


        x=rearrange(
            x,
            "b (d h w) c -> b c d h w",
            d=D,
            h=H,
            w=W
        )


        x=self.up1(x)


        if x.shape[2:]!=x3.shape[2:]:

            x=F.interpolate(
                x,
                size=x3.shape[2:],
                mode="trilinear",
                align_corners=False
            )


        x=torch.cat(
            [
                x,
                x3
            ],
            dim=1
        )


        x=self.dec1(x)


        x=F.interpolate(
            x,
            scale_factor=2,
            mode="trilinear",
            align_corners=False
        )


        x=self.dec2(x)


        x=F.interpolate(
            x,
            scale_factor=2,
            mode="trilinear",
            align_corners=False
        )


        x=self.dec3(x)


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
# MRI PROCESS
# =========================


def normalize(img):

    img = img.astype(
        np.float32
    )

    mask = img > 0


    if mask.sum() > 0:

        img[mask] = (
            img[mask]
            -
            img[mask].mean()
        ) / (
            img[mask].std()
            +
            1e-8
        )


    return img





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
# SLIDING WINDOW
# =========================


def sliding_window_inference(
        model,
        volume,
        patch_size=80,
        stride=40
):


    _,C,D,H,W = volume.shape


    output = torch.zeros(
        (1,4,D,H,W),
        device=volume.device
    )


    count = torch.zeros(
        (1,1,D,H,W),
        device=volume.device
    )



    for z in range(
        0,
        D-patch_size+1,
        stride
    ):

        for y in range(
            0,
            H-patch_size+1,
            stride
        ):

            for x in range(
                0,
                W-patch_size+1,
                stride
            ):


                patch = volume[
                    :,
                    :,
                    z:z+patch_size,
                    y:y+patch_size,
                    x:x+patch_size
                ]



                with torch.no_grad():

                    pred = model(
                        patch
                    )



                output[
                    :,
                    :,
                    z:z+patch_size,
                    y:y+patch_size,
                    x:x+patch_size
                ] += pred



                count[
                    :,
                    :,
                    z:z+patch_size,
                    y:y+patch_size,
                    x:x+patch_size
                ] += 1



    output = output / count



    return torch.argmax(
        output,
        dim=1
    )[0].cpu().numpy()





# =========================
# STREAMLIT UI
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


    if st.button(
        "Predict"
    ):



        if len(files)!=4:

            st.error(
                "Cần upload đúng 4 file MRI"
            )

            st.stop()



        volumes={}



        for f in files:


            name=f.name.lower()


            data=read_nii(f)



            if "t1ce" in name or "t1c" in name:

                volumes["t1c"]=data



            elif "_t1" in name or "t1n" in name:

                volumes["t1n"]=data



            elif "_t2" in name and "flair" not in name:

                volumes["t2w"]=data



            elif "flair" in name:

                volumes["t2f"]=data




        try:


            img=np.stack(
                [
                    volumes["t1n"],
                    volumes["t1c"],
                    volumes["t2w"],
                    volumes["t2f"]
                ],
                axis=0
            )


        except:


            st.error(
                "Sai thứ tự hoặc thiếu MRI"
            )

            st.stop()




        # tensor

        x=torch.tensor(
            img,
            dtype=torch.float32
        )


        x=x.unsqueeze(0)


        x=x.to(
            DEVICE
        )



        # =====================
        # PREDICT
        # =====================


        with st.spinner(
            "Running TransUNet inference..."
        ):


            mask = sliding_window_inference(
                model,
                x
            )



        st.success(
            "Segmentation completed!"
        )



        # =====================
        # LOAD GROUND TRUTH
        # =====================


        gt_mask=None


        seg_path=(
            "datasets/BraTS2021_00495_seg.nii"
        )


        if os.path.exists(seg_path):

            gt_mask=nib.load(
                seg_path
            ).get_fdata()



        # =====================
        # FIND BEST SLICE
        # =====================


        slice_id=np.argmax(
            [
                (mask[:,:,i]>0).sum()
                for i in range(
                    mask.shape[2]
                )
            ]
        )



        # =====================
        # DISPLAY
        # =====================


        fig,ax=plt.subplots(
            1,
            3,
            figsize=(15,5)
        )



        ax[0].imshow(
            img[0,:,:,slice_id],
            cmap="gray"
        )

        ax[0].set_title(
            "MRI"
        )



        if gt_mask is not None:


            ax[1].imshow(
                img[0,:,:,slice_id],
                cmap="gray"
            )


            ax[1].imshow(
                gt_mask[:,:,slice_id],
                cmap="jet",
                alpha=0.5
            )


            ax[1].set_title(
                "Ground Truth"
            )



        ax[2].imshow(
            img[0,:,:,slice_id],
            cmap="gray"
        )


        ax[2].imshow(
            mask[:,:,slice_id],
            cmap="jet",
            alpha=0.5
        )


        ax[2].set_title(
            "AI Prediction"
        )



        st.pyplot(fig)