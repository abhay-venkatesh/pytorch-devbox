class TrainerBase:
    def load_checkpoint(checkpoint_path):
        epochs_done = 0
        if checkpoint_path:
            epochs_done = int(checkpoint_path.split('.')[1].split("/")[4])
            num_epochs -= epochs_done
            self.model.load_state_dict(torch.load(checkpoint_path))
        return epochs_done

    def write_checkpoint(self, epoch):
        checkpoint_filename = str(epoch + 1) + ".ckpt"
        checkpoint_path = (
            self.experiment_root + "checkpoints/" + checkpoint_filename)
        if not os.path.exists(self.experiment_root + "checkpoints/"):
            os.makedirs(self.experiment_root + "checkpoints/")
        torch.save(self.model.state_dict(), checkpoint_path)
