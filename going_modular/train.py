from pathlib import Path
import torch 
import datasetup, engine, utils
from torch import nn
import json

from torchvision.models import EfficientNet_B0_Weights
from torchvision.transforms import v2
from modelbuilder import PlantNet


NUM_EPOCHS = 10
BATCH_SIZE = 32
NUM_WORKERS = 4

weights = EfficientNet_B0_Weights.DEFAULT


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data" / "New Plant Diseases Dataset(Augmented)"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224,224)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])

    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224)),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])

    train_dataloader, test_dataloader, class_names = datasetup.create_dataloaders(
        data_root=data_root,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = PlantNet(output_shape=len(class_names)).to(device)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.model.features[-2:].parameters():
        param.requires_grad = True

    for param in model.model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    
    
    optimizer = torch.optim.AdamW(
        [{"params":model.model.features[-2:].parameters(),
            "lr": 1e-5},
        {"params":model.model.classifier.parameters(),
            "lr":1e-3}],
        weight_decay=1e-4)
    
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    results = engine.train(
        model=model,
        train_data_loader=train_dataloader,
        test_data_loader=test_dataloader,
        loss_fn=criterion,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device,
        scheduler=scheduler,
    )

    utils.save_model(
        model=model,
        target_dir=str(project_root / "models"),
        model_name="PlantNet_feature_extractor.pth",
    )
    with open(project_root / "models" / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)

if __name__ == "__main__":
    main()
