from c4service import ModelService, MainInterface

if __name__ == "__main__":
    service = ModelService("checkpoints/pytorch_model.bin")
    ui = MainInterface(service)
    ui.render()
