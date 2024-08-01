import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_CONTAM2_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "CONTAM-2"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(1, 1, 1, 1, 1), (0.03326209446380745, 0.9980069071951875, 0.9012693649772621, 0.7880875551892006, 0.9467520953445779), (0.5586587815082229, 0.9848623184606812, 0.6708517455349591, 0.7924703773655554, 0.6446912384349336), (0.20506481515557543, 0.7982577744493301, 0.8046505840884806, 0.9233324616339879, 0.7252097422824303), (0.16707844979882186, 0.8670421935023686, 0.7435598696257112, 0.7425617141306486, 0.8992466493610534), (0.11935584988119473, 0.7380309655588216, 0.7859860873979299, 0.8617839960500299, 0.8710377624202182), (0.08857969740046587, 0.6197503839871101, 0.8498400358405728, 0.8730402261932305, 0.7883400965888846), (0.08857969740046587, 0.6197503839871101, 0.8498400358405728, 0.8730402261932305, 0.7883400965888846)], [(1, 1, 1, 1, 1), (0.4025584488949174, 0.7805415192508689, 0.9456483676784814, 0.982317050295404, 0.8975987021579711), (0.17406964190455285, 0.7718715038032441, 0.9319150868892526, 0.8981351942783503, 0.7945680197491656), (0.17406964190455285, 0.7718715038032441, 0.9319150868892526, 0.8981351942783503, 0.7945680197491656)], [(1, 1, 1, 1, 1), (1, 1, 1, 1, 1)], [(1, 1, 1, 1, 1), (1, 1, 1, 1, 1)], [(1, 1, 1, 1, 1), (1, 1, 1, 1, 1)], [(1, 1, 1, 1, 1), (0.8420821309911748, 0.7750584253138287, 0.6871811414448725, 0.8463317337532065, 0.8028906113471015), (0.6066071342617003, 0.8107883747769479, 0.7804731839193084, 0.8536461618632083, 0.7855266435978799), (0.13917897198098392, 0.917781073576413, 0.788741525741815, 0.8139095756442267, 0.9133195916587663), (0.10283039984962046, 0.973400427835828, 0.6092631508891321, 0.9601696726668841, 0.8431757421653147), (0.10283039984962046, 0.973400427835828, 0.6092631508891321, 0.9601696726668841, 0.8431757421653147)], [(1, 1, 1, 1, 1), (0.15532444168522114, 0.8048985780726431, 0.9763238320768246, 0.7371860820182379, 0.8733646698901084), (0.15265302098165928, 0.8318161377724625, 0.9754705994617858, 0.695260625475601, 0.7859104761084028), (0.15265302098165928, 0.8318161377724625, 0.9754705994617858, 0.695260625475601, 0.7859104761084028)], [(1, 1, 1, 1, 1), (0.25482071354116975, 0.8143541534397901, 0.89158782536403, 0.9921094680574652, 0.8520586530743632), (0.33400845119584305, 0.8158981571222695, 0.7327990199956569, 0.9490010925550544, 0.6155955477254172), (0.33400845119584305, 0.8158981571222695, 0.7327990199956569, 0.9490010925550544, 0.6155955477254172)], [(1, 1, 1, 1, 1), (0.7007941377268129, 0.9827784801409404, 0.46256385888282275, 0.7666013702873805, 0.9127250697572751), (0.1427668749111495, 0.8876552981399704, 0.8522797872485118, 0.6898583284324343, 0.8831793891036216), (0.13590465329311974, 0.8833809140471812, 0.5627404810045893, 0.8196071736696857, 0.8298511464635466), (0.13590465329311974, 0.8833809140471812, 0.5627404810045893, 0.8196071736696857, 0.8298511464635466)], [(1, 1, 1, 1, 1), (0.040689602136480915, 0.931800068778548, 0.6429362743466034, 0.8138730023721198, 0.8935573834134117), (0.040689602136480915, 0.931800068778548, 0.6429362743466034, 0.8138730023721198, 0.8935573834134117)], [(1, 1, 1, 1, 1), (0.5862691092640103, 0.7403456661831352, 0.8294184276645615, 0.9558004261475264, 0.9481644866099147), (0.7555080263748928, 0.9722994575831777, 0.5803289275403174, 0.8165744451916509, 0.7021602152495936), (0.6066672488084967, 0.9833847513757713, 0.9222966779111208, 0.5582866482724489, 0.6844865345333702), (0.15021666284768515, 0.785221256391616, 0.8714823274100955, 0.8286853042353279, 0.9026664839486194), (0.4808178732660873, 0.6793355325473918, 0.8969729907276068, 0.748339036399154, 0.5741538159605102), (0.2291514276674709, 0.8309396851890383, 0.7844304570838658, 0.8387099302959782, 0.5593482082114619), (0.2291514276674709, 0.8309396851890383, 0.7844304570838658, 0.8387099302959782, 0.5593482082114619)], [(1, 1, 1, 1, 1), (0.9198270189398947, 0.9905537602107931, 0.76553103379683, 0.9139358971963327, 0.9811970335638578), (0.5520363978165135, 0.6842077503249078, 0.8581209318919932, 0.942878328291395, 0.8488451693113417), (0.5520363978165135, 0.6842077503249078, 0.8581209318919932, 0.942878328291395, 0.8488451693113417)], [(1, 1, 1, 1, 1), (0.4263700183213139, 0.7997073741488006, 0.980172989395443, 0.6602657826932341, 0.8379112024059358), (0.058815362219138856, 0.7891364817368771, 0.8877973599969063, 0.9468879231607281, 0.8606820635082828), (0.07712701592650716, 0.9485294481958555, 0.8146345281144562, 0.8472250083514494, 0.8024362479119421), (0.07712701592650716, 0.9485294481958555, 0.8146345281144562, 0.8472250083514494, 0.8024362479119421)], [(1, 1, 1, 1, 1), (1, 1, 1, 1, 1)], [(1, 1, 1, 1, 1), (0.5031508644240396, 0.8726218683890414, 0.9605216341532069, 0.9334303764983822, 0.7860903922717092), (0.2191723477067073, 0.9051112537884948, 0.7954301250748025, 0.8817953959604359, 0.9971882615739384), (0.34391985124333974, 0.8327705862969818, 0.9937024574470966, 0.829713395931848, 0.7316305011462291), (0.09649355315385831, 0.8878790332653651, 0.8742893861728238, 0.7736830338663587, 0.8597082967914934), (0.09649355315385831, 0.8878790332653651, 0.8742893861728238, 0.7736830338663587, 0.8597082967914934)], [(1, 1, 1, 1, 1), (0.6503781511631458, 0.8532174118955671, 0.7719697448354464, 0.9916025335558987, 0.7775535038046374), (0.6503781511631458, 0.8532174118955671, 0.7719697448354464, 0.9916025335558987, 0.7775535038046374)], [(1, 1, 1, 1, 1), (0.7597550989661973, 0.8912192551357684, 0.8134746142203734, 0.6872769235525281, 0.9785142937980065), (0.5445257404961983, 0.9353164812889482, 0.8482775630992216, 0.3566835530079387, 0.8597025509965911), (0.10688795760103856, 0.9230293191480681, 0.9525507036900489, 0.7980502373525988, 0.6937771500799915), (0.6847509440100278, 0.7246671676940217, 0.9805386764817045, 0.1039997284840661, 0.8713043747077022), (0.06202868626033114, 0.669292282129823, 0.9304743983174384, 0.07288056150059141, 0.9674631606862736), (0.15482425717717166, 0.6472971263895283, 0.8804336639889987, 0.16113553068511896, 0.8446649775584963), (0.15482425717717166, 0.6472971263895283, 0.8804336639889987, 0.16113553068511896, 0.8446649775584963)], [(1, 1, 1, 1, 1), (0.3957520132689781, 0.9249243853577116, 0.938991519229076, 0.9410429358801178, 0.7898275657287179), (0.33500083807859904, 0.9148561531421915, 0.6162404267066179, 0.96469541887209, 0.9387694036737169), (0.33500083807859904, 0.9148561531421915, 0.6162404267066179, 0.96469541887209, 0.9387694036737169)], [(1, 1, 1, 1, 1), (0.5085550143335581, 0.9081904699335849, 0.6229174091869084, 0.7470346927603745, 0.9596493029052054), (0.1674222731552629, 0.8981191699413554, 0.5426603802184945, 0.8745330986340728, 0.8991886423973455), (0.1674222731552629, 0.8981191699413554, 0.5426603802184945, 0.8745330986340728, 0.8991886423973455)], [(1, 1, 1, 1, 1), (0.34943611865926366, 0.9013212915684164, 0.7887688928898261, 0.7414765949889859, 0.9341714292084001), (0.04559761110793406, 0.7954171040204255, 0.876628287681072, 0.7284137912816528, 0.9599871350632339), (0.04559761110793406, 0.7954171040204255, 0.876628287681072, 0.7284137912816528, 0.9599871350632339)], [(1, 1, 1, 1, 1), (0.40892757709532424, 0.9097821235737489, 0.7130336743572272, 0.8813449331372386, 0.7483233322043096), (0.011274649841973365, 0.7066279877858753, 0.9141519668380752, 0.8824353829367458, 0.8764191736223149), (0.011274649841973365, 0.7066279877858753, 0.9141519668380752, 0.8824353829367458, 0.8764191736223149)], [(1, 1, 1, 1, 1), (0.9282241040073833, 0.8602272544352499, 0.800498179277317, 0.9540242265064827, 0.8681639215392284), (0.9641822435776487, 0.8395958714270828, 0.6774208498894091, 0.9874919809862814, 0.8150910273052133), (0.3695690354496146, 0.7132962649608086, 0.6773164062485594, 0.8314306908141784, 0.9968977245396765), (0.10544438984534543, 0.8339484300141394, 0.817191020114285, 0.9704861873437481, 0.7668817640541604), (0.0604524238906112, 0.7353588070149143, 0.8796726665394183, 0.8453299742724363, 0.8137262061366465), (0.0604524238906112, 0.7353588070149143, 0.8796726665394183, 0.8453299742724363, 0.8137262061366465)], [(1, 1, 1, 1, 1), (0.40049100138762234, 0.6494683458212335, 0.8485009461846662, 0.9587593065625839, 0.619415900399561), (0.40049100138762234, 0.6494683458212335, 0.8485009461846662, 0.9587593065625839, 0.619415900399561)], [(1, 1, 1, 1, 1), (0.15162631998264106, 0.9244822923308977, 0.9077056215151142, 0.9737547749050413, 0.750296473983132), (0.15162631998264106, 0.9244822923308977, 0.9077056215151142, 0.9737547749050413, 0.750296473983132)]]"
        self.expected_all_intermediate_budgets = "[[0, 30, 440, 2020, 6060, 8550, 9720, 10000], [0, 1840, 3540, 10000], [0, 10000], [0, 10000], [0, 10000], [0, 3720, 4920, 5150, 8440, 10000], [0, 910, 4040, 10000], [0, 400, 640, 10000], [0, 280, 3210, 7010, 10000], [0, 70, 10000], [0, 280, 1500, 2920, 3570, 4370, 7460, 10000], [0, 3990, 4130, 10000], [0, 420, 2780, 8670, 10000], [0, 10000], [0, 1580, 3240, 6350, 8200, 10000], [0, 1990, 10000], [0, 80, 190, 1240, 2400, 3650, 9730, 10000], [0, 500, 5840, 10000], [0, 1280, 5060, 10000], [0, 2410, 8000, 10000], [0, 1950, 6640, 10000], [0, 140, 400, 3470, 5260, 8090, 10000], [0, 780, 10000], [0, 510, 10000]]"
        self.expected_all_est_objectives = "[[5.0, 3.6673780171700368, 3.6515344613043514, 3.4565153776098043, 3.419488876418604, 3.3761946613081957, 3.219550440010264, 3.219550440010264], [5.0, 4.008664088277642, 3.5705594466245647, 3.5705594466245647], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 3.9535440428501833, 3.837041498419045, 3.5729307386022047, 3.4888393934067783, 3.4888393934067783], [5.0, 3.547097603743034, 3.4411108597999123, 3.4411108597999123], [5.0, 3.8049308134768176, 3.447302268594242, 3.447302268594242], [5.0, 3.8254629167952316, 3.455739677835687, 3.2314843684781214, 3.2314843684781214], [5.0, 3.3228563310471633, 3.3228563310471633], [5.0, 4.059998115869148, 3.826871071939633, 3.755121860901207, 3.5382720348333443, 3.37961924890075, 3.2425797084478143, 3.2425797084478143], [5.0, 4.571044743707708, 3.886088577636151, 3.886088577636151], [5.0, 3.7044273669647283, 3.5433191906219337, 3.4899522485002104, 3.4899522485002104], [5.0, 5.0], [5.0, 4.05581513573638, 3.7986973841043787, 3.7317367920654956, 3.4920533032498984, 3.4920533032498984], [5.0, 4.044721345254694, 4.044721345254694], [5.0, 4.130240185672874, 3.544505888888899, 3.4742953678717465, 3.3652608913775235, 2.702139088894458, 2.6883555557993146, 2.6883555557993146], [5.0, 3.990538419464601, 3.769562240473214, 3.769562240473214], [5.0, 3.7463468891196303, 3.38192356434653, 3.38192356434653], [5.0, 3.715174327314892, 3.4060439291543183, 3.4060439291543183], [5.0, 3.6614116403678474, 3.3909091610249855, 3.3909091610249855], [5.0, 4.41113768576566, 4.283781973185635, 3.588510122012837, 3.493951791371678, 3.3345400778540273, 3.3345400778540273], [5.0, 3.4766355003556675, 3.4766355003556675], [5.0, 3.7078654827168265, 3.7078654827168265]]"
        self.expected_objective_curves = "[([0, 30, 440, 2020, 6060, 8550, 9720, 10000], [5.0, 3.6673780171700368, 3.6515344613043514, 3.4565153776098043, 3.419488876418604, 3.3761946613081957, 3.219550440010264, 3.219550440010264]), ([0, 1840, 3540, 10000], [5.0, 4.008664088277642, 3.5705594466245647, 3.5705594466245647]), ([0, 10000], [5.0, 5.0]), ([0, 10000], [5.0, 5.0]), ([0, 10000], [5.0, 5.0]), ([0, 3720, 4920, 5150, 8440, 10000], [5.0, 3.9535440428501833, 3.837041498419045, 3.5729307386022047, 3.4888393934067783, 3.4888393934067783]), ([0, 910, 4040, 10000], [5.0, 3.547097603743034, 3.4411108597999123, 3.4411108597999123]), ([0, 400, 640, 10000], [5.0, 3.8049308134768176, 3.447302268594242, 3.447302268594242]), ([0, 280, 3210, 7010, 10000], [5.0, 3.8254629167952316, 3.455739677835687, 3.2314843684781214, 3.2314843684781214]), ([0, 70, 10000], [5.0, 3.3228563310471633, 3.3228563310471633]), ([0, 280, 1500, 2920, 3570, 4370, 7460, 10000], [5.0, 4.059998115869148, 3.826871071939633, 3.755121860901207, 3.5382720348333443, 3.37961924890075, 3.2425797084478143, 3.2425797084478143]), ([0, 3990, 4130, 10000], [5.0, 4.571044743707708, 3.886088577636151, 3.886088577636151]), ([0, 420, 2780, 8670, 10000], [5.0, 3.7044273669647283, 3.5433191906219337, 3.4899522485002104, 3.4899522485002104]), ([0, 10000], [5.0, 5.0]), ([0, 1580, 3240, 6350, 8200, 10000], [5.0, 4.05581513573638, 3.7986973841043787, 3.7317367920654956, 3.4920533032498984, 3.4920533032498984]), ([0, 1990, 10000], [5.0, 4.044721345254694, 4.044721345254694]), ([0, 80, 190, 1240, 2400, 3650, 9730, 10000], [5.0, 4.130240185672874, 3.544505888888899, 3.4742953678717465, 3.3652608913775235, 2.702139088894458, 2.6883555557993146, 2.6883555557993146]), ([0, 500, 5840, 10000], [5.0, 3.990538419464601, 3.769562240473214, 3.769562240473214]), ([0, 1280, 5060, 10000], [5.0, 3.7463468891196303, 3.38192356434653, 3.38192356434653]), ([0, 2410, 8000, 10000], [5.0, 3.715174327314892, 3.4060439291543183, 3.4060439291543183]), ([0, 1950, 6640, 10000], [5.0, 3.6614116403678474, 3.3909091610249855, 3.3909091610249855]), ([0, 140, 400, 3470, 5260, 8090, 10000], [5.0, 4.41113768576566, 4.283781973185635, 3.588510122012837, 3.493951791371678, 3.3345400778540273, 3.3345400778540273]), ([0, 780, 10000], [5.0, 3.4766355003556675, 3.4766355003556675]), ([0, 510, 10000], [5.0, 3.7078654827168265, 3.7078654827168265])]"
        self.expected_progress_curves = "[([0.0, 0.003, 0.044, 0.202, 0.606, 0.855, 0.972, 1.0], [1.0, 0.4235177532716309, 0.4166639501682026, 0.3323001613581201, 0.31628277543006794, 0.29755402360189537, 0.22979091163590462, 0.22979091163590462]), ([0.0, 0.184, 0.354, 1.0], [1.0, 0.5711555407193514, 0.38163476785474953, 0.38163476785474953]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.372, 0.492, 0.515, 0.844, 1.0], [1.0, 0.5473110236415888, 0.4969128991707546, 0.3826605709290886, 0.34628328747341286, 0.34628328747341286]), ([0.0, 0.091, 0.404, 1.0], [1.0, 0.37148535108765535, 0.32563628281549306, 0.32563628281549306]), ([0.0, 0.04, 0.064, 1.0], [1.0, 0.4830220583787009, 0.3283146396925043, 0.3283146396925043]), ([0.0, 0.028, 0.321, 0.701, 1.0], [1.0, 0.4919040918462282, 0.3319645994701043, 0.23495343933249582, 0.23495343933249582]), ([0.0, 0.007, 1.0], [1.0, 0.27448026310432216, 0.27448026310432216]), ([0.0, 0.028, 0.15, 0.292, 0.357, 0.437, 0.746, 1.0], [1.0, 0.5933622549570406, 0.4925132491705451, 0.4614750801223483, 0.3676674763570362, 0.2990354744370988, 0.23975319995205313, 0.23975319995205313]), ([0.0, 0.399, 0.413, 1.0], [1.0, 0.814437182427242, 0.5181302967424929, 0.5181302967424929]), ([0.0, 0.042, 0.278, 0.867, 1.0], [1.0, 0.4395450233337024, 0.3698508379900293, 0.34676470021671946, 0.34676470021671946]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.158, 0.324, 0.635, 0.82, 1.0], [1.0, 0.5915527292130354, 0.4803255237156488, 0.4513588752300351, 0.3476736006987806, 0.3476736006987806]), ([0.0, 0.199, 1.0], [1.0, 0.5867536388903356, 0.5867536388903356]), ([0.0, 0.008, 0.019, 0.124, 0.24, 0.365, 0.973, 1.0], [1.0, 0.623748446042761, 0.37036419473481, 0.3399916514168736, 0.2928241569660024, 0.005962652746931969, 0.0, 0.0]), ([0.0, 0.05, 0.584, 1.0], [1.0, 0.5633145127193434, 0.46772187971483487, 0.46772187971483487]), ([0.0, 0.128, 0.506, 1.0], [1.0, 0.4576790933287949, 0.3000323039675055, 0.3000323039675055]), ([0.0, 0.241, 0.8, 1.0], [1.0, 0.44419407755011736, 0.3104665923669607, 0.3104665923669607]), ([0.0, 0.195, 0.664, 1.0], [1.0, 0.4209367435418874, 0.3039194055072765, 0.3039194055072765]), ([0.0, 0.014, 0.04, 0.347, 0.526, 0.809, 1.0], [1.0, 0.7452625918697653, 0.6901694684876084, 0.3894000950153801, 0.3484948723811721, 0.2795345640960579, 0.2795345640960579]), ([0.0, 0.078, 1.0], [1.0, 0.3410039751285896, 0.3410039751285896]), ([0.0, 0.051, 1.0], [1.0, 0.4410323263489751, 0.4410323263489751])]"

        # Convert the expected values from string to their actual types
        self.expected_all_recommended_xs = eval(self.expected_all_recommended_xs, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_all_intermediate_budgets = eval(self.expected_all_intermediate_budgets, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_all_est_objectives = eval(self.expected_all_est_objectives, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_objective_curves = eval(self.expected_objective_curves, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_progress_curves = eval(self.expected_progress_curves, {'nan': float('nan'), 'inf': float('inf')})
        
        # Number of macro-replications and post-replications
        self.num_macroreps = 24
        self.num_postreps = 200

        # Setup the solver and experiment
        self.myexperiment = ProblemSolver(self.expected_solver_name, self.expected_problem_name)
        self.assertEqual(self.myexperiment.solver.name, self.expected_solver_name, "Solver name does not match (expected: " + self.expected_solver_name + ", actual: " + self.myexperiment.solver.name + ")")
        self.assertEqual(self.myexperiment.problem.name, self.expected_problem_name, "Problem name does not match (expected: " + self.expected_problem_name + ", actual: " + self.myexperiment.problem.name + ")")

    def test_run(self):
        # Check actual run results against expected
        self.myexperiment.run(n_macroreps=self.num_macroreps)
        self.assertEqual(self.myexperiment.n_macroreps, self.num_macroreps, "Number of macro-replications for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_recommended_xs[mrep]), len(self.expected_all_recommended_xs[mrep]), "Length of recommended solutions for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
            # For each list of recommended solutions
            for list in range(len(self.myexperiment.all_recommended_xs[mrep])):
                # Check to make sure the tuples are the same length
                self.assertEqual(len(self.myexperiment.all_recommended_xs[mrep][list]), len(self.expected_all_recommended_xs[mrep][list]), "Recommended solutions for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")
                # For each tuple of recommended solutions
                for tuple in range(len(self.myexperiment.all_recommended_xs[mrep][list])):
                    self.assertAlmostEqual(self.myexperiment.all_recommended_xs[mrep][list][tuple], self.expected_all_recommended_xs[mrep][list][tuple], 5, "Recommended solutions for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + " and tuple " + str(tuple) + ".")
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_intermediate_budgets[mrep]), len(self.expected_all_intermediate_budgets[mrep]), "Length of intermediate budgets for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
            # For each list of intermediate budgets
            for list in range(len(self.myexperiment.all_intermediate_budgets[mrep])):
                # Check the values in the list
                self.assertAlmostEqual(self.myexperiment.all_intermediate_budgets[mrep][list], self.expected_all_intermediate_budgets[mrep][list], 5, "Intermediate budgets for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")
            
    def test_post_replicate(self):
        # Simulate results from the run method
        self.myexperiment = ProblemSolver(self.expected_solver_name, self.expected_problem_name)
        self.myexperiment.n_macroreps = self.num_macroreps
        self.myexperiment.all_recommended_xs = self.expected_all_recommended_xs
        self.myexperiment.all_intermediate_budgets = self.expected_all_intermediate_budgets

        # Check actual post-replication results against expected
        self.myexperiment.post_replicate(n_postreps=self.num_postreps)
        self.assertEqual(self.myexperiment.n_postreps, self.num_postreps, "Number of post-replications for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_est_objectives[mrep]), len(self.expected_all_est_objectives[mrep]), "Estimated objectives for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
            # For each list in the estimated objectives
            for list in range(len(self.myexperiment.all_est_objectives[mrep])):
                # Check the values in the list
                self.assertAlmostEqual(self.myexperiment.all_est_objectives[mrep][list], self.expected_all_est_objectives[mrep][list], 5, "Estimated objectives for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")

    def test_post_normalize(self):
        # Simulate results from the post_replicate method
        self.myexperiment = ProblemSolver(self.expected_solver_name, self.expected_problem_name)
        self.myexperiment.n_macroreps = self.num_macroreps
        self.myexperiment.n_postreps = self.num_postreps
        self.myexperiment.all_recommended_xs = self.expected_all_recommended_xs
        self.myexperiment.all_intermediate_budgets = self.expected_all_intermediate_budgets
        self.myexperiment.all_est_objectives = self.expected_all_est_objectives

        # Check actual post-normalization results against expected
        post_normalize([self.myexperiment], n_postreps_init_opt=self.num_postreps)

        # Loop through each curve object and convert it into a tuple
        for i in range(len(self.myexperiment.objective_curves)):
            self.myexperiment.objective_curves[i] = (self.myexperiment.objective_curves[i].x_vals, self.myexperiment.objective_curves[i].y_vals)
        for i in range(len(self.myexperiment.progress_curves)):
            self.myexperiment.progress_curves[i] = (self.myexperiment.progress_curves[i].x_vals, self.myexperiment.progress_curves[i].y_vals)

        for mrep in range(self.num_macroreps):
            # Check to make sure the same number of objective curves are present
            # This should probably always be 2 (x and y)
            self.assertEqual(len(self.myexperiment.objective_curves[mrep]), len(self.expected_objective_curves[mrep]), "Number of objective curves for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
            # Make sure that curves are only checked if they exist
            if (len(self.myexperiment.objective_curves[mrep]) > 0):
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(len(self.myexperiment.objective_curves[mrep][0]), len(self.expected_objective_curves[mrep][0]), "Length of X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                self.assertEqual(len(self.myexperiment.objective_curves[mrep][1]), len(self.expected_objective_curves[mrep][1]), "Length of Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                # Check X (0) and Y (1) values
                for x_index in range(len(self.myexperiment.objective_curves[mrep][0])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.objective_curves[mrep][0][x_index])):
                        self.assertTrue(math.isnan(self.expected_objective_curves[mrep][0][x_index]), "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.objective_curves[mrep][0][x_index], self.expected_objective_curves[mrep][0][x_index], 5, "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                for y_index in range(len(self.myexperiment.objective_curves[mrep][1])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.objective_curves[mrep][1][y_index])):
                        self.assertTrue(math.isnan(self.expected_objective_curves[mrep][1][y_index]), "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.objective_curves[mrep][1][y_index], self.expected_objective_curves[mrep][1][y_index], 5, "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
            
            # Check to make sure the same number of progress curves are present
            # This should probably always be 2 (x and y)
            self.assertEqual(len(self.myexperiment.progress_curves[mrep]), len(self.expected_progress_curves[mrep]), "Number of progress curves for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
            # Make sure that curves are only checked if they exist
            if (len(self.myexperiment.progress_curves[mrep]) > 0):
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(len(self.myexperiment.progress_curves[mrep][0]), len(self.expected_progress_curves[mrep][0]), "Length of X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                self.assertEqual(len(self.myexperiment.progress_curves[mrep][1]), len(self.expected_progress_curves[mrep][1]), "Length of Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                # Check X (0) and Y (1) values
                for x_index in range(len(self.myexperiment.progress_curves[mrep][0])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.progress_curves[mrep][0][x_index])):
                        self.assertTrue(math.isnan(self.expected_progress_curves[mrep][0][x_index]), "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.progress_curves[mrep][0][x_index], self.expected_progress_curves[mrep][0][x_index], 5, "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                for y_index in range(len(self.myexperiment.progress_curves[mrep][1])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.progress_curves[mrep][1][y_index])):
                        self.assertTrue(math.isnan(self.expected_progress_curves[mrep][1][y_index]), "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.progress_curves[mrep][1][y_index], self.expected_progress_curves[mrep][1][y_index], 5, "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")      
