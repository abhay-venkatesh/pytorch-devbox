class TrainerBase:
    def write_checkpoint(self, epoch):
        checkpoint_filename = str(epoch) + ".ckpt"
        checkpoint_path = (
            self.experiment_root + "checkpoints/" + checkpoint_filename)
        if not os.path.exists(self.experiment_root + "checkpoints/"):
            os.makedirs(self.experiment_root + "checkpoints/") 
        torch.save(self.model.state_dict(), checkpoint_path)