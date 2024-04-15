
from uagents.setup import fund_agent_if_low
from uagents import Agent, Context, Protocol, Model
from ai_engine import UAgentResponse, UAgentResponseType
import os
from pydantic import Field
import random
import requests
import string
import boto3
from typing import List
import time
import argparse

from diffusers import (
    StableCascadeDecoderPipeline,
    StableCascadePriorPipeline,
    StableCascadeUNet,
    DiffusionPipeline,
)
import torch

from config import *


class NFTGenerator(Model):
    prompt: str = Field(description="Describe how your NFT collection should look like")
    amount_of_images: int = Field(description="How many images should the service create?")


def get_agent_seed() -> str:
    return AGENT_SEED if AGENT_SEED else ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(32))


def get_agent_mailbox_key(adress: str) -> str:
    if MAILBOX_KEY:
        return MAILBOX_KEY
    res = requests.post(
        "https://agentverse.ai/v1/agents",
        json={
            'address': adress,
            'name': 'queed'
        },
        headers={
            "Authorization": f"bearer {ACCESS_TOKEN_FETCH_AI}"
        },
    )
    print(f"Create mailbox response {res}, {res.content}")
    if res.status_code != 201:
        raise RuntimeError(f"Failed to create a malibox: {res.content}")
    data = res.json()
    return data["key"]


def register_agent(protocol: Protocol, seed: str) -> None:
    data = {
        "agent": Agent(seed=seed).address,
        "name": AGENT_NAME,
        "description": "Generates images for NFT collections.",
        "protocolDigest": protocol.digest,
        "modelDigest": NFTGenerator.build_schema_digest(NFTGenerator),
        "modelName": 'Queed/NFTGenerator',
        "fields": [
            {
                "name": "prompt",
                "required": True,
                "field_type": "str",
                "description": "Describe how your NFT collection should look like"
            },
            {
                "name": "amount_of_images",
                "required": True,
                "field_type": "int",
                "description": "How many images should the service create?",
            },
        ],
        "taskType": "task",
    }

    res = requests.post("https://agentverse.ai/v1beta1/services", json=data, headers={
        "Authorization": f"bearer {ACCESS_TOKEN_FETCH_AI}"
    })
    if res.status_code != 200:
        raise RuntimeError(f"Failed to register agent in agentverse. {res}, {res.content}")
    print(f"registering service in agentverse: {res}")
    print(res.content)

def generate_images_stable_diffusion(pipe: DiffusionPipeline, prompt: str, number_of_images: int) -> List[str]:
    files = []
    for i in range(number_of_images):
        strength = random.uniform(0.1, 1.0)
        guidance_scale = random.uniform(1, 30)
        result = pipe(prompt=prompt, strength=strength, guidance_scale=guidance_scale)
        image = result.images[0]
        file_name = f"image{i}.png"
        image.save(file_name)
        files.append(file_name)
    return files

def generate_images_stable_cascade(
    prior: StableCascadePriorPipeline,
    decoder: StableCascadeDecoderPipeline,
    prompt: str,
    number_of_images: int,
):
    prior_output = prior(
        prompt=prompt,
        height=512,
        width=512,
        negative_prompt="",
        guidance_scale=4.0,
        num_images_per_prompt=number_of_images,
        num_inference_steps=10,
    )
    
    output = decoder(
        image_embeddings=prior_output.image_embeddings,
        prompt=prompt,
        negative_prompt="",
        guidance_scale=0.0,
        output_type="pil",
        num_inference_steps=5,
    ).images

    files = []
    for i, image in enumerate(output):
        file_name = f"image{i}.png"
        image.save(file_name)
        files.append(file_name)
    return files

def main():

    assert ACCESS_TOKEN_FETCH_AI, "FETCH_AI access token was not specified"
    assert AWS_ACCESS_KEY_ID
    assert AWS_SECRET_ACCESS_KEY
    assert BUCKET_NAME

    parser = argparse.ArgumentParser(
        prog='ImageGeneratingAgent',
        description='Registeres a fetch AI agent that ',
    )
    parser.add_argument("--model")
    args = parser.parse_args()

    seed_phrase = get_agent_seed()

    agent_mailbox_key = get_agent_mailbox_key(adress=Agent(seed=seed_phrase).address)
    
    nft_generation_agent = Agent(
        name=AGENT_NAME,
        seed=seed_phrase,
        mailbox=f"{agent_mailbox_key}@https://agentverse.ai",
    )

    nft_generator_protocol = Protocol("NFTGenerator")

    if args.model == "gpu":
        model = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipe.to("cuda")
    elif args.model == "cpu":
        prior_unet = StableCascadeUNet.from_pretrained(
            "stabilityai/stable-cascade-prior", subfolder="prior_lite"
        )
        decoder_unet = StableCascadeUNet.from_pretrained(
            "stabilityai/stable-cascade", subfolder="decoder_lite"
        )

        prior = StableCascadePriorPipeline.from_pretrained(
            "stabilityai/stable-cascade-prior", prior=prior_unet
        )
        decoder = StableCascadeDecoderPipeline.from_pretrained(
            "stabilityai/stable-cascade", decoder=decoder_unet
        )
        prior.to("cpu")
        decoder.to("cpu")
    else:
        raise RuntimeError("Not supported argument. The model can be cpu (cascade) or gpu (stable diffusion)!")

    @nft_generator_protocol.on_message(model=NFTGenerator, replies={UAgentResponse})
    async def answer(ctx: Context, sender: str, msg: NFTGenerator):
        images_file_paths = []
        if args.model == "cpu":
            images_file_paths = generate_images_stable_cascade(
                prior=prior,
                decoder=decoder,
                number_of_images=msg.amount_of_images,
                prompt=msg.prompt,
            )
        elif args.model == "gpu":
            images_file_paths = generate_images_stable_diffusion(
                number_of_images=msg.amount_of_images,
                pipe=pipe,
                prompt=msg.prompt,
            )
        bucket_folder = time.time()
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        s3 = session.resource('s3')
        for path in images_file_paths:
            s3.meta.client.upload_file(
                Filename=f"{path}",
                Bucket=BUCKET_NAME,
                Key=f"{bucket_folder}/{path}",
            )
        await ctx.send(
            sender, UAgentResponse(message=f"Link to your images: https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{bucket_folder}", type=UAgentResponseType.FINAL)
        )

    nft_generation_agent.include(nft_generator_protocol, publish_manifest=True)
    fund_agent_if_low(nft_generation_agent.wallet.address())

    register_agent(protocol=nft_generator_protocol, seed=seed_phrase)

    nft_generation_agent.run()


if __name__ == "__main__":
    main()
