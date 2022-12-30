class GanCriterion:
    """Criterion for discriminator and generator"""
    def __init__(self, discriminator_loss, generator_loss):
        self._discriminator_loss = discriminator_loss
        self._generator_loss = generator_loss

    def get_discriminator_loss(self):
        return self._discriminator_loss

    def get_generator_loss(self):
        return self._generator_loss
