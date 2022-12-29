class GanOptimizer:
    def __init__(self, discriminator_opt, generator_opt):
        self._discriminator_opt = discriminator_opt
        self._generator_opt = generator_opt

    def get_discriminator_optimizer(self):
        return self._discriminator_opt

    def get_generator_optimizer(self):
        return self._generator_opt
