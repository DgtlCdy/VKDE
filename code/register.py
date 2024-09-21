import world
import dataloader
import model
import utils
from pprint import pprint


if world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
else:
    dataset = dataloader.Loader(path="C:/codes/VKDE/data/"+world.dataset)

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print('===========end===================')

MODELS = {
    'VKDE': model.VKDE, 
    'MultVAE': model.MultVAE
}
