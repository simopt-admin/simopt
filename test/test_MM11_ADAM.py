import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_MM11_ADAM(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "MM1-1"
        self.expected_solver_name = "ADAM"
        self.expected_all_recommended_xs = "[[(5,), (4.500000005366897,), (4.003004560324342,), (3.5123680289566863,), (3.033809830342704,), (2.7040480453692344,), (2.841378779661173,), (2.841378779661173,)], [(5,), (4.500000005377763,), (4.003063613212165,), (3.512698763886283,), (3.0349726782257616,), (2.8392016843721914,), (2.8392016843721914,)], [(5,), (4.500000005375356,), (4.003021944059995,), (3.512528800915692,), (3.03440158276607,), (2.7037094220298337,), (2.8428708705357058,), (2.8428708705357058,)], [(5,), (4.50000000538071,), (4.003092018928721,), (3.512809639456232,), (3.035197131139862,), (2.872500415114987,), (2.8857068749001282,), (2.8857068749001282,)], [(5,), (4.500000005399897,), (4.003102733658987,), (3.5128936519448044,), (3.0354861202091614,), (2.7346144472620924,), (2.8758091523718488,), (2.8758091523718488,)], [(5,), (4.500000005388163,), (4.003058472726469,), (3.5126780187496123,), (3.0349493734745496,), (2.686477889201981,), (2.831428970373492,), (2.831428970373492,)], [(5,), (4.50000000536467,), (4.003002495195689,), (3.5123763945691593,), (3.033685402293352,), (2.577816876095712,), (2.6278661195002955,), (2.774297909432402,), (2.7587563844921235,), (2.7587563844921235,)], [(5,), (4.500000005356673,), (4.002969569343121,), (3.512261056149438,), (3.0334024427985837,), (2.577291436683673,), (2.6015203636852884,), (2.753612451462658,), (2.7617575873358655,), (2.7617575873358655,)], [(5,), (4.500000005375804,), (4.003037907171925,), (3.512604308481135,), (3.0346011103126522,), (2.6880116603792086,), (2.834310483874307,), (2.834310483874307,)], [(5,), (4.500000005357146,), (4.002973607736823,), (3.512297952023572,), (3.0336148245939447,), (2.6621236542037057,), (2.8068534006951555,), (2.8068534006951555,)], [(5,), (4.5000000053752816,), (4.003015900513098,), (3.512549978005597,), (3.034471734054668,), (2.72024470111445,), (2.858299103088228,), (2.858299103088228,)], [(5,), (4.500000005389652,), (4.003088733239444,), (3.51283359501668,), (3.0356056701242027,), (2.994066506634766,), (2.994066506634766,)], [(5,), (4.500000005353427,), (4.002969951949881,), (3.5122040555942666,), (3.0332524999350734,), (2.576994459945876,), (2.644935391731322,), (2.7906674536264062,), (2.74711831751059,), (2.74711831751059,)], [(5,), (4.50000000538435,), (4.003083892552371,), (3.512799271992763,), (3.0353348848491284,), (2.7418959021334945,), (2.879815457671644,), (2.879815457671644,)], [(5,), (4.500000005346948,), (4.0029371483642695,), (3.5120916699474916,), (3.0328826159576128,), (2.5759422670842334,), (2.6031027469943058,), (2.752029836112292,), (2.747932132046137,), (2.747932132046137,)], [(5,), (4.500000005384983,), (4.0030885828004035,), (3.5128121301898196,), (3.0352909624561812,), (2.7299369709013073,), (2.8727309139217514,), (2.8727309139217514,)], [(5,), (4.500000005366741,), (4.003012045838118,), (3.5124421157538506,), (3.034041012392123,), (2.6742872575151817,), (2.820308168145676,), (2.820308168145676,)], [(5,), (4.500000005387876,), (4.003066959772039,), (3.512696015232104,), (3.034878418332684,), (2.7038038020591633,), (2.8480762049081454,), (2.8480762049081454,)], [(5,), (4.500000005396141,), (4.003128283261345,), (3.5130358427924624,), (3.0360830512889847,), (2.9077430024251147,), (2.8988017461488655,), (2.8988017461488655,)], [(5,), (4.500000005368764,), (4.0030339578350835,), (3.5125786862682524,), (3.0346408506618205,), (2.7078848366806616,), (2.85030843089649,), (2.85030843089649,)], [(5,), (4.500000005365071,), (4.003017617919233,), (3.5124752239027575,), (3.0342017427214723,), (2.6985830842364833,), (2.8401069890222446,), (2.8401069890222446,)], [(5,), (4.500000005397168,), (4.003101344250994,), (3.5129051754447262,), (3.0358894676844215,), (2.899043599444121,), (2.90450265461085,), (2.90450265461085,)], [(5,), (4.500000005375175,), (4.003038432447738,), (3.512582586384228,), (3.0345779509277597,), (2.710443791943149,), (2.8530979440564868,), (2.8530979440564868,)], [(5,), (4.500000005380378,), (4.003055601102281,), (3.512681977648549,), (3.0348960652909365,), (2.702920700757322,), (2.8449456763771552,), (2.8449456763771552,)]]"
        self.expected_all_intermediate_budgets = "[[0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 450, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 450, 660, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 180, 420, 450, 690, 1000], [0, 60, 90, 120, 150, 180, 420, 450, 690, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 630, 1000], [0, 60, 90, 120, 150, 180, 420, 450, 690, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 180, 420, 450, 690, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 450, 660, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 450, 660, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 420, 450, 1000]]"
        self.expected_all_est_objectives = "[[2.7852321578182706, 2.3577222923567365, 2.0007947090608287, 1.7277595315142087, 1.5670475808635072, 1.5536274548033373, 1.5455346393846563, 1.5455346393846563], [2.7857037031168543, 2.358264583497712, 2.001600818463035, 1.729985862726112, 1.5689877932554779, 1.5478331564209389, 1.5478331564209389], [2.7866293625352507, 2.35980239336231, 2.0046780793778654, 1.7358468907673676, 1.5821271702679323, 1.576026391369124, 1.5649350507059543, 1.5649350507059543], [2.7889080044387127, 2.362302779280393, 2.0072424575131644, 1.7377589470462727, 1.5827974447301625, 1.5655141984346137, 1.5659938798921655, 1.5659938798921655], [2.7833651638972787, 2.3549506392236124, 1.9973644207333896, 1.72438765925803, 1.5637066548134007, 1.546201397075579, 1.5435656983349022, 1.5435656983349022], [2.787955763524055, 2.361043297403672, 2.005396134757599, 1.7361086135219, 1.5820419866805162, 1.5844296210523061, 1.567191382599853, 1.567191382599853], [2.7843462630059106, 2.356852454095064, 2.000138788267423, 1.7282668447527374, 1.5675295393684667, 1.58063903766362, 1.5663710685177055, 1.5463057856633415, 1.547127331680449, 1.547127331680449], [2.7907221687784363, 2.3649491744090856, 2.0117334854436093, 1.7452409673170575, 1.594053693287159, 1.625785476715929, 1.6172073691478284, 1.5841956544451268, 1.583280184592205, 1.583280184592205], [2.789502875694011, 2.362519357738991, 2.006812063682465, 1.7369781349621172, 1.580988291079133, 1.5752024844909869, 1.5625408955175755, 1.5625408955175755], [2.7891645344327056, 2.3629750270472005, 2.008371136708882, 1.7398882929189385, 1.5874299336702535, 1.5938290757196913, 1.5732166386438513, 1.5732166386438513], [2.7863020842335002, 2.359153090948389, 2.002682752124033, 1.7311827933000592, 1.5719278198506097, 1.5533772497762737, 1.5503896192431876, 1.5503896192431876], [2.781108319206661, 2.352845331808715, 1.9947292015577047, 1.7215067108493571, 1.5606928046414312, 1.553321297101712, 1.553321297101712], [2.781564747274972, 2.353374528977135, 1.9957622543451214, 1.7230660180785404, 1.5606797693656853, 1.5683839900946799, 1.5518290237164805, 1.5372310118101198, 1.5389697527620685, 1.5389697527620685], [2.7819442310007103, 2.3534103370104646, 1.994756069783606, 1.7197804227545914, 1.5553613143546696, 1.5327694988253229, 1.5335174232328188, 1.5335174232328188], [2.784695397913865, 2.3574441909455595, 2.000637170394297, 1.7282427792526298, 1.567129793519485, 1.5900036285495331, 1.580732110111496, 1.551422348393338, 1.5517945972855625, 1.5517945972855625], [2.782112928233372, 2.3537491279224176, 1.9956362256294056, 1.7217209204154984, 1.5596562873161595, 1.5412520679766288, 1.5387102248581122, 1.5387102248581122], [2.784512429482461, 2.357322031758369, 2.001260024456027, 1.7295120992979582, 1.5698101801855493, 1.5670342128063783, 1.5508285054830566, 1.5508285054830566], [2.783456075233837, 2.3551108515756005, 1.9975407923659947, 1.7245489390744282, 1.5626772602945176, 1.5496956015673578, 1.542112690724275, 1.542112690724275], [2.7872953386099404, 2.3603197679753536, 2.004473168542364, 1.7339836538748663, 1.5748938697945576, 1.5580624990912662, 1.557439686031052, 1.557439686031052], [2.7844968268172887, 2.356219675032622, 1.9985954638959469, 1.72570913385646, 1.5656332049441903, 1.548374525180978, 1.5433886059623387, 1.5433886059623387], [2.781707203439503, 2.3530775294907325, 1.994674288101445, 1.7198580002974746, 1.5558658976164255, 1.5382622371666366, 1.5327240790548555, 1.5327240790548555], [2.7902297278963424, 2.3640397696265656, 2.0092440735732238, 1.7400500342910812, 1.5843340056626942, 1.5677653265964704, 1.5680855818751047, 1.5680855818751047], [2.7850791792196157, 2.357583506730152, 2.0012636055810837, 1.7307624938626904, 1.5762739240335712, 1.5705846700090753, 1.5594178398020904, 1.5594178398020904], [2.7868278653888137, 2.3592736155091045, 2.003013377840716, 1.731450123543994, 1.5727221644427045, 1.5640922283947942, 1.5537426619738612, 1.5537426619738612]]"
        self.expected_objective_curves = "[([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.3577222923567365, 2.0007947090608287, 1.7277595315142087, 1.5670475808635072, 1.5536274548033373, 1.5455346393846563, 1.5455346393846563]), ([0, 60, 90, 120, 150, 450, 1000], [2.7854035060729516, 2.358264583497712, 2.001600818463035, 1.729985862726112, 1.5689877932554779, 1.5478331564209389, 1.5478331564209389]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.35980239336231, 2.0046780793778654, 1.7358468907673676, 1.5821271702679323, 1.576026391369124, 1.5649350507059543, 1.5649350507059543]), ([0, 60, 90, 120, 150, 450, 660, 1000], [2.7854035060729516, 2.362302779280393, 2.0072424575131644, 1.7377589470462727, 1.5827974447301625, 1.5655141984346137, 1.5659938798921655, 1.5659938798921655]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.3549506392236124, 1.9973644207333896, 1.72438765925803, 1.5637066548134007, 1.546201397075579, 1.5435656983349022, 1.5435656983349022]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.361043297403672, 2.005396134757599, 1.7361086135219, 1.5820419866805162, 1.5844296210523061, 1.567191382599853, 1.567191382599853]), ([0, 60, 90, 120, 150, 180, 420, 450, 690, 1000], [2.7854035060729516, 2.356852454095064, 2.000138788267423, 1.7282668447527374, 1.5675295393684667, 1.58063903766362, 1.5663710685177055, 1.5463057856633415, 1.547127331680449, 1.547127331680449]), ([0, 60, 90, 120, 150, 180, 420, 450, 690, 1000], [2.7854035060729516, 2.3649491744090856, 2.0117334854436093, 1.7452409673170575, 1.594053693287159, 1.625785476715929, 1.6172073691478284, 1.5841956544451268, 1.583280184592205, 1.583280184592205]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.362519357738991, 2.006812063682465, 1.7369781349621172, 1.580988291079133, 1.5752024844909869, 1.5625408955175755, 1.5625408955175755]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.3629750270472005, 2.008371136708882, 1.7398882929189385, 1.5874299336702535, 1.5938290757196913, 1.5732166386438513, 1.5732166386438513]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.359153090948389, 2.002682752124033, 1.7311827933000592, 1.5719278198506097, 1.5533772497762737, 1.5503896192431876, 1.5503896192431876]), ([0, 60, 90, 120, 150, 630, 1000], [2.7854035060729516, 2.352845331808715, 1.9947292015577047, 1.7215067108493571, 1.5606928046414312, 1.553321297101712, 1.553321297101712]), ([0, 60, 90, 120, 150, 180, 420, 450, 690, 1000], [2.7854035060729516, 2.353374528977135, 1.9957622543451214, 1.7230660180785404, 1.5606797693656853, 1.5683839900946799, 1.5518290237164805, 1.5372310118101198, 1.5389697527620685, 1.5389697527620685]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.3534103370104646, 1.994756069783606, 1.7197804227545914, 1.5553613143546696, 1.5327694988253229, 1.5335174232328188, 1.5335174232328188]), ([0, 60, 90, 120, 150, 180, 420, 450, 690, 1000], [2.7854035060729516, 2.3574441909455595, 2.000637170394297, 1.7282427792526298, 1.567129793519485, 1.5900036285495331, 1.580732110111496, 1.551422348393338, 1.5517945972855625, 1.5517945972855625]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.3537491279224176, 1.9956362256294056, 1.7217209204154984, 1.5596562873161595, 1.5412520679766288, 1.5387102248581122, 1.5387102248581122]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.357322031758369, 2.001260024456027, 1.7295120992979582, 1.5698101801855493, 1.5670342128063783, 1.5508285054830566, 1.5508285054830566]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.3551108515756005, 1.9975407923659947, 1.7245489390744282, 1.5626772602945176, 1.5496956015673578, 1.542112690724275, 1.542112690724275]), ([0, 60, 90, 120, 150, 450, 660, 1000], [2.7854035060729516, 2.3603197679753536, 2.004473168542364, 1.7339836538748663, 1.5748938697945576, 1.5580624990912662, 1.557439686031052, 1.557439686031052]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.356219675032622, 1.9985954638959469, 1.72570913385646, 1.5656332049441903, 1.548374525180978, 1.5433886059623387, 1.5433886059623387]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.3530775294907325, 1.994674288101445, 1.7198580002974746, 1.5558658976164255, 1.5382622371666366, 1.5524654089524859, 1.5524654089524859]), ([0, 60, 90, 120, 150, 450, 660, 1000], [2.7854035060729516, 2.3640397696265656, 2.0092440735732238, 1.7400500342910812, 1.5843340056626942, 1.5677653265964704, 1.5680855818751047, 1.5680855818751047]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.357583506730152, 2.0012636055810837, 1.7307624938626904, 1.5762739240335712, 1.5705846700090753, 1.5594178398020904, 1.5594178398020904]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.3592736155091045, 2.003013377840716, 1.731450123543994, 1.5727221644427045, 1.5640922283947942, 1.5537426619738612, 1.5537426619738612])]"
        self.expected_progress_curves = "[([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.653120286642884, 0.36362677181881115, 0.14217593159877476, 0.011827172787569822, 0.0009425013742096301, -0.005621344318921135, -0.005621344318921135]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.45, 1.0], [1.0, 0.6535601231133785, 0.3642805835584994, 0.14398164367556343, 0.01340082226478373, -0.0037570844330024594, -0.0037570844330024594]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6548073956797705, 0.3667764598089108, 0.14873535195576348, 0.024057786343630407, 0.019109623160858506, 0.01011376141477934, 0.01011376141477934]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.45, 0.66, 1.0], [1.0, 0.6568353855066097, 0.3688563518499534, 0.15028616483385576, 0.02460142633966561, 0.010583491184677777, 0.01097254677365839, 0.01097254677365839]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6508722799184611, 0.36084456536785425, 0.13944110471326146, 0.009117445463944046, -0.0050805566731504815, -0.007218294769517612, -0.007218294769517612]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6558138566239655, 0.3673588535084897, 0.14894762762081395, 0.02398869642937194, 0.0259252367774772, 0.011943806166554227, 0.011943806166554227]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.18, 0.42, 0.45, 0.69, 1.0], [1.0, 0.6524147862907546, 0.3630947736633989, 0.14258739851646796, 0.012218075223049102, 0.022850805548902838, 0.011278473426765186, -0.004995890145279929, -0.004329558219106008, -0.004329558219106008]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.18, 0.42, 0.45, 0.69, 1.0], [1.0, 0.6589817991301918, 0.37249889314292967, 0.15635461246172866, 0.03373104005124246, 0.05946776073728485, 0.05251030878723574, 0.025735473311066546, 0.02499296251100305, 0.02499296251100305]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6570110459546924, 0.36850727201236505, 0.1496528710083352, 0.02313407477087664, 0.018441376409410654, 0.008171932223216207, 0.008171932223216207]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6573806259922251, 0.3697717905068929, 0.15201321494094466, 0.02835870251671802, 0.033548859317276816, 0.016830714972495428, 0.016830714972495428]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6542807654982249, 0.36515810828056355, 0.14495243902753013, 0.01578539177561178, 0.0007395674007619867, -0.001683612270677859, -0.001683612270677859]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.63, 1.0], [1.0, 0.6491647267008143, 0.35870721623261426, 0.13710445179013306, 0.006672999810907379, 0.0006941858242722369, 0.0006941858242722369]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.18, 0.42, 0.45, 0.69, 1.0], [1.0, 0.6495939430334557, 0.35954509510895805, 0.13836916023966922, 0.006662427280318565, 0.012911095195591703, -0.0005161534366499668, -0.012356173580771086, -0.010945931691085348, -0.010945931691085348]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6496229858811163, 0.3587290082641558, 0.1357043100483883, 0.0023487841027438174, -0.015974776165294025, -0.015368156571623654, -0.015368156571623654]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.18, 0.42, 0.45, 0.69, 1.0], [1.0, 0.6528947267288653, 0.3634989968178606, 0.14256787969377624, 0.011893853066303898, 0.030446151096083427, 0.022926293886957615, -0.0008459958870472879, -0.0005440757070367794, -0.0005440757070367794]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.649897769272711, 0.3594428769067545, 0.137278190898886, 0.0058323109493233615, -0.009094812628505719, -0.011156427177081332, -0.011156427177081332]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.65279564698798, 0.36400417551514014, 0.143597388026995, 0.01406783623084748, 0.011816330347742522, -0.0013276444886018515, -0.0013276444886018515]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6510022234674213, 0.3609876152363083, 0.13957191405135783, 0.00828253370212306, -0.002246509692252199, -0.008396786710046235, -0.008396786710046235]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.45, 0.66, 1.0], [1.0, 0.6552270230838161, 0.3666102626283874, 0.14722413505294174, 0.018191068062908863, 0.0045396359734948, 0.004034490531344257, 0.004034490531344257]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6519015577159218, 0.3618430284418986, 0.14051291407783242, 0.010680013881035831, -0.003317996078685691, -0.007361929209054415, -0.007361929209054415]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6493530554438061, 0.3586626775356691, 0.13576723092257034, 0.0027580368161884737, -0.011519776880137682, 0.0, 0.0]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.45, 0.66, 1.0], [1.0, 0.6582442075311944, 0.3704798040449453, 0.15214439863339477, 0.025847685933817487, 0.012409315341717136, 0.012669065023702216, 0.012669065023702216]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6530077216836955, 0.36400708006084714, 0.144611546456889, 0.019310389659213483, 0.014696002255836765, 0.005638913150499567, 0.005638913150499567]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6543785194414253, 0.3654262691212866, 0.14516926276309255, 0.016429661422198268, 0.009430172909299237, 0.0010359425378763229, 0.0010359425378763229])]"

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
        self.myexperiment.has_run = True
        self.myexperiment.has_postreplicated= True
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
