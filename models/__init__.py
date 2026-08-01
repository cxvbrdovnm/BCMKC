# from models.sgmg import build
#from models.sgmg_losh import build
#from models.sgmg_rpe import build
# from models.sgmg_san import build
# from models.sgmg_clip import build
#from models.sgmg_san_inDecoder import build
#from models.sgmg_maple import build
#from models.sgmg_soc import build
#from models.sgmg_bike import build
#from models.sgmg_san_laff import build
#from models.sgmg_san_bat_BEMSeg import build
#from models.sgmg_san_ffm2 import build
from models.sgmg_bike import build
# from models.sgmg_bike_san import build
# from models.sgmg_bike_spacy import build
# from models.sgmg_bike_dshmp import build
# from models.sgmg_bike_san import build
# from models.sgmg_mtcm import build

def build_model(args):
    print("\n **** BUILD MODEL FOR SgMg. ****  \n")
    return build(args)
