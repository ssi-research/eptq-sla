import torchvision
import datasets
import helpers
import compression

if __name__ == '__main__':
    val_dataloader = datasets.get_imagenet_dataset_loader(batch_size=32)
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).to(
        helpers.get_device())
    model.eval()
    cc = compression.CompressionConfig()
    qm = compression.QuantizationModelWrapper(model, cc)

    helpers.classification_evaluation(qm, val_dataloader)
    qm.apply_quantization()
    cc.enable_weights_quantization()
    helpers.classification_evaluation(qm, val_dataloader)
