import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_PARAMESTI1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "PARAMESTI-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(1, 1), (2.4113681969383234, 4.318464516345556), (1.9268503221182314, 5.518432243083116), (1.8335257714086588, 4.421207443720463), (1.8335257714086588, 4.421207443720463)], [(1, 1), (2.7671853413978944, 8.793537060533582), (1.8238893372167329, 4.647975737037825), (1.8238893372167329, 4.647975737037825)], [(1, 1), (2.998069279500845, 4.170376173532159), (1.0691241703410232, 8.57228763192795), (1.2508345745442417, 9.870697642375044), (1.5137935275605543, 8.200691640526957), (2.0673546697501495, 4.902396919950507), (1.6343786298182692, 6.755515651578842), (2.1040986388112692, 5.250778883849738), (2.110124972883145, 5.3495508527631355), (2.110124972883145, 5.3495508527631355)], [(1, 1), (1.0155504388116514, 1.9046428891517508), (2.3057927322352514, 1.5023677912290445), (2.3112392099894947, 6.447737434699523), (2.316519693782576, 5.50220301529817), (1.801236898721493, 4.157777887889604), (1.801236898721493, 4.157777887889604)], [(1, 1), (1.2858672849508925, 0.9037923294791963), (3.6408893489290466, 6.190815109989965), (2.5317221076456367, 8.10740335244683), (2.6145498441826485, 5.925052174276406), (1.8603758927802991, 4.217514228784237), (1.9703032648710255, 5.769749555133262), (1.9703032648710255, 5.769749555133262)], [(1, 1), (3.3183198378445877, 5.472627408804023), (0.9640260783064701, 7.31223616037637), (2.40758780168329, 2.6346541621973896), (2.469907451546926, 7.628763803835697), (1.5567598637207534, 8.460953527055294), (1.5533405918848804, 7.833058065170441), (2.3404783429856173, 5.7314743128713825), (2.0422424300542166, 5.6575772077255095), (2.0422424300542166, 5.6575772077255095)], [(1, 1), (4.527815174005357, 6.971020440727531), (2.418986025254497, 1.196418280865765), (3.625650479093963, 6.459370152919785), (2.5605649494793057, 6.228412361864414), (2.1456882812555795, 3.9313017447736964), (2.1456882812555795, 3.9313017447736964)], [(1, 1), (0.6578676336995485, 2.8637598553584076), (3.7938722500636777, 9.122615619097848), (1.2569842204574306, 2.970262204579669), (1.8969748412889356, 3.0981233762366847), (2.395060817332159, 4.66957205526321), (2.0889145155889497, 6.162766039873319), (2.0889145155889497, 6.162766039873319)], [(1, 1), (1.7218048328849038, 9.602482721795424), (1.797354556887818, 6.867724390697357), (2.414283761212375, 7.282372642618957), (2.420635437823872, 5.960116426624409), (2.420635437823872, 5.960116426624409)], [(1, 1), (1.2989873956398523, 7.993845434677753), (2.502384693058212, 4.267474222866501), (1.5866377463845194, 3.466395597534787), (1.5866377463845194, 3.466395597534787)], [(1, 1), (3.098752410230344, 5.202075450269435), (2.537534441032253, 5.708536876662558), (2.537534441032253, 5.708536876662558)], [(1, 1), (2.216092582011423, 6.560204119776016), (2.095414437480784, 6.101279607593585), (2.6467486549922548, 6.256366040889206), (2.8264725208064276, 4.996360621401809), (2.8264725208064276, 4.996360621401809)], [(1, 1), (2.4549789692591006, 2.9196003377616573), (1.5145520992639565, 8.82555869110306), (1.7674790483982405, 4.685056380134013), (1.6874368520655825, 5.896627687289976), (1.6874368520655825, 5.896627687289976)], [(1, 1), (1.3837202502213917, 2.325472644716108), (2.508982239609656, 5.411816945406126), (1.7574672895374701, 6.01930329371595), (1.902126893783546, 4.1489854034709195), (1.902126893783546, 4.1489854034709195)], [(1, 1), (1.003106059587109, 6.664167321973203), (2.1775586884543783, 6.131532960142674), (2.2266035697733852, 5.2479114346358875), (2.4832339355984376, 5.093790515747021), (2.4832339355984376, 5.093790515747021)], [(1, 1), (3.5003201079709885, 7.023965338800287), (1.8685829888249894, 4.259523443128186), (1.8159988730279184, 6.320446228504371), (1.8159988730279184, 6.320446228504371)], [(1, 1), (1.4881333353770279, 5.001637198180085), (2.410985480059167, 5.644568694748518), (2.410985480059167, 5.644568694748518)], [(1, 1), (1.8850609908794722, 7.320598465177343), (1.8592316491576342, 5.647829537710302), (1.8592316491576342, 5.647829537710302)], [(1, 1), (1.6811146590094663, 9.414030266813537), (2.4074846357472253, 7.740056498961466), (2.2458717105540718, 6.715758127085327), (2.148029131719391, 5.63582773682479), (2.148029131719391, 5.63582773682479)], [(1, 1), (1.4021929504480526, 7.639789708488681), (1.677956811412009, 5.118859543167704), (1.6813499972971158, 5.14215527222685)], [(1, 1), (1.672969847004332, 6.906170405234082), (2.5776593447786627, 3.8948277907735163), (2.001772540358056, 5.806207032825579), (2.001772540358056, 5.806207032825579)], [(1, 1), (0.5064081318054562, 5.475664251888675), (3.463217295158934, 5.908283706992635), (1.8396974164659772, 6.380828547620294), (1.9035449096554293, 5.170140550818582), (1.9035449096554293, 5.170140550818582)], [(1, 1), (3.842403000504669, 7.75122173075893), (3.293940079291244, 1.4343743652454275), (1.3405186218740126, 3.811915206322066), (1.5860865945243308, 4.749264632548915), (2.252158210321541, 3.8104894432662535), (2.252158210321541, 3.8104894432662535)], [(1, 1), (2.4815650617157887, 7.202992845494886), (1.2605466711320237, 6.0901071103621), (1.446435647261938, 5.351554441410891), (2.114753782741909, 5.873113939889637), (1.793361744498658, 4.648785706154869), (1.793361744498658, 4.648785706154869)]]"
        self.expected_all_intermediate_budgets = "[[0, 30, 70, 140, 1000], [0, 30, 40, 1000], [0, 30, 70, 230, 260, 330, 490, 920, 950, 1000], [0, 20, 140, 150, 280, 700, 1000], [0, 20, 40, 70, 160, 290, 580, 1000], [0, 30, 90, 120, 130, 170, 240, 270, 280, 1000], [0, 30, 60, 90, 150, 420, 1000], [0, 20, 30, 40, 60, 310, 660, 1000], [0, 20, 240, 780, 960, 1000], [0, 70, 100, 740, 1000], [0, 20, 40, 1000], [0, 20, 440, 960, 970, 1000], [0, 20, 220, 250, 290, 1000], [0, 50, 100, 260, 560, 1000], [0, 20, 90, 270, 870, 1000], [0, 80, 180, 290, 1000], [0, 60, 550, 1000], [0, 30, 700, 1000], [0, 20, 60, 200, 400, 1000], [0, 30, 40, 1000], [0, 40, 430, 540, 1000], [0, 20, 70, 130, 370, 1000], [0, 140, 160, 200, 320, 600, 1000], [0, 30, 130, 150, 390, 720, 1000]]"
        self.expected_all_est_objectives = "[[-9.16944227603262, -4.941528873034248, -4.6862774991901714, -4.735757520942245, -4.735757520942245], [-8.940090362495347, -6.6007493728513325, -4.598011859328733, -4.598011859328733], [-9.121210005202611, -5.914961707825279, -6.942962358768506, -7.236688866043877, -5.719948285836633, -4.595121421429469, -4.98937255604907, -4.601642669629433, -4.607175485483224, -4.607175485483224], [-8.779886386724968, -7.340511963803717, -6.60081656477736, -4.924010768184911, -4.730199362237067, -4.633244397622134, -4.633244397622134], [-8.99288952739613, -8.311376570591937, -7.1405261639904305, -5.8169172388750185, -5.005684837558, -4.651329021400779, -4.669237067305324, -4.669237067305324], [-8.87740808504234, -6.391672077886543, -6.8876892167042865, -5.423872134921744, -5.532768062446162, -6.02336562143931, -5.675018136344402, -4.758261957483205, -4.614467756646956, -4.614467756646956], [-9.024638576352391, -11.131246586030965, -7.292809573682223, -7.647058165148387, -5.182785644192512, -4.788250138222532, -4.788250138222532], [-8.921050660074993, -8.273808602003013, -9.610166400917251, -5.871695536397843, -5.066615392689244, -4.8544681821601925, -4.756050363500685, -4.756050363500685], [-8.550164686658025, -6.899540350240091, -5.1027912222963, -5.373080078441992, -4.840691519451912, -4.840691519451912], [-8.983830735669818, -6.146117067967971, -4.944586417648114, -5.091884989354981, -5.091884989354981], [-9.025843710458552, -5.980583405101112, -4.992319917623182, -4.992319917623182], [-9.203733926294058, -4.77217737303398, -4.629949173794984, -5.087861980602993, -5.23161083810709, -5.23161083810709], [-9.33623207280299, -5.36625241182486, -6.262771084577963, -4.72683255888335, -4.85868878328377, -4.85868878328377], [-9.573886675373538, -6.585066784671261, -5.324747253205283, -5.005374099782757, -5.040552233423579, -5.040552233423579], [-8.941889405024408, -6.353871356585989, -4.851187833411799, -4.761766291534669, -5.024945769183457, -5.024945769183457], [-9.587195496910567, -7.165460619279511, -4.796964736748716, -4.827976154006989, -4.827976154006989], [-9.346621843523279, -5.151768028087783, -5.00133548585014, -5.00133548585014], [-9.398264139884379, -5.2465850180658125, -4.8019564927438605, -4.8019564927438605], [-8.499265659696825, -6.487693440944086, -5.5584982918247015, -4.953608510569406, -4.588506273756264, -4.588506273756264], [-9.254478807791063, -5.8921907776201214, -4.85093694728059, -4.848155249986304], [-9.605611629911163, -5.133767943371997, -5.323383995037183, -4.7379852619515015, -4.7379852619515015], [-8.501695309087717, -8.421685871782755, -7.066793467266279, -4.811448206558395, -4.536076301914798, -4.536076301914798], [-9.152042163756049, -8.868412917034739, -8.713579572038785, -5.412677553292934, -4.819255057071357, -4.892908965909024, -4.892908965909024], [-9.213076761398039, -5.309082459177796, -5.628389879434484, -5.121232857832066, -4.695369803419064, -4.706450569866729, -4.706450569866729]]"
        self.expected_objective_curves = "[([0, 30, 70, 140, 1000], [-9.265122221743944, -4.941528873034248, -4.6862774991901714, -4.735757520942245, -4.735757520942245]), ([0, 30, 40, 1000], [-9.265122221743944, -6.6007493728513325, -4.598011859328733, -4.598011859328733]), ([0, 30, 70, 230, 260, 330, 490, 920, 950, 1000], [-9.265122221743944, -5.914961707825279, -6.942962358768506, -7.236688866043877, -5.719948285836633, -4.595121421429469, -4.98937255604907, -4.601642669629433, -4.607175485483224, -4.607175485483224]), ([0, 20, 140, 150, 280, 700, 1000], [-9.265122221743944, -7.340511963803717, -6.60081656477736, -4.924010768184911, -4.730199362237067, -4.633244397622134, -4.633244397622134]), ([0, 20, 40, 70, 160, 290, 580, 1000], [-9.265122221743944, -8.311376570591937, -7.1405261639904305, -5.8169172388750185, -5.005684837558, -4.651329021400779, -4.669237067305324, -4.669237067305324]), ([0, 30, 90, 120, 130, 170, 240, 270, 280, 1000], [-9.265122221743944, -6.391672077886543, -6.8876892167042865, -5.423872134921744, -5.532768062446162, -6.02336562143931, -5.675018136344402, -4.758261957483205, -4.614467756646956, -4.614467756646956]), ([0, 30, 60, 90, 150, 420, 1000], [-9.265122221743944, -11.131246586030965, -7.292809573682223, -7.647058165148387, -5.182785644192512, -4.788250138222532, -4.788250138222532]), ([0, 20, 30, 40, 60, 310, 660, 1000], [-9.265122221743944, -8.273808602003013, -9.610166400917251, -5.871695536397843, -5.066615392689244, -4.8544681821601925, -4.756050363500685, -4.756050363500685]), ([0, 20, 240, 780, 960, 1000], [-9.265122221743944, -6.899540350240091, -5.1027912222963, -5.373080078441992, -4.840691519451912, -4.840691519451912]), ([0, 70, 100, 740, 1000], [-9.265122221743944, -6.146117067967971, -4.944586417648114, -5.091884989354981, -5.091884989354981]), ([0, 20, 40, 1000], [-9.265122221743944, -5.980583405101112, -4.992319917623182, -4.992319917623182]), ([0, 20, 440, 960, 970, 1000], [-9.265122221743944, -4.77217737303398, -4.629949173794984, -5.087861980602993, -5.23161083810709, -5.23161083810709]), ([0, 20, 220, 250, 290, 1000], [-9.265122221743944, -5.36625241182486, -6.262771084577963, -4.72683255888335, -4.85868878328377, -4.85868878328377]), ([0, 50, 100, 260, 560, 1000], [-9.265122221743944, -6.585066784671261, -5.324747253205283, -5.005374099782757, -5.040552233423579, -5.040552233423579]), ([0, 20, 90, 270, 870, 1000], [-9.265122221743944, -6.353871356585989, -4.851187833411799, -4.761766291534669, -5.024945769183457, -5.024945769183457]), ([0, 80, 180, 290, 1000], [-9.265122221743944, -7.165460619279511, -4.796964736748716, -4.827976154006989, -4.827976154006989]), ([0, 60, 550, 1000], [-9.265122221743944, -5.151768028087783, -5.00133548585014, -5.00133548585014]), ([0, 30, 700, 1000], [-9.265122221743944, -5.2465850180658125, -4.8019564927438605, -4.8019564927438605]), ([0, 20, 60, 200, 400, 1000], [-9.265122221743944, -6.487693440944086, -5.5584982918247015, -4.953608510569406, -4.588506273756264, -4.588506273756264]), ([0, 30, 40, 1000], [-9.265122221743944, -5.8921907776201214, -4.85093694728059, -4.848155249986304]), ([0, 40, 430, 540, 1000], [-9.265122221743944, -5.133767943371997, -5.323383995037183, -4.7379852619515015, -4.7379852619515015]), ([0, 20, 70, 130, 370, 1000], [-9.265122221743944, -8.421685871782755, -7.066793467266279, -4.811448206558395, -4.536076301914798, -4.536076301914798]), ([0, 140, 160, 200, 320, 600, 1000], [-9.265122221743944, -8.868412917034739, -8.713579572038785, -5.412677553292934, -4.819255057071357, -4.892908965909024, -4.892908965909024]), ([0, 30, 130, 150, 390, 720, 1000], [-9.265122221743944, -5.309082459177796, -5.628389879434484, -5.121232857832066, -4.695369803419064, -4.706450569866729, -4.706450569866729])]"
        self.expected_progress_curves = "[([0.0, 0.03, 0.07, 0.14, 1.0], [1.0, 0.07178791429389822, 0.016989187636159794, 0.027611822673051553, 0.027611822673051553]), ([0.0, 0.03, 0.04, 1.0], [1.0, 0.4279982228422625, -0.0019601508108566703, -0.0019601508108566703]), ([0.0, 0.03, 0.07, 0.23, 0.26, 0.33, 0.49, 0.92, 0.95, 1.0], [1.0, 0.28076966835868233, 0.5014663323046589, 0.5645251058654529, 0.23890314656407693, -0.0025806854390452126, 0.08205925140109924, -0.0011806690857722772, 7.145338312734482e-06, 7.145338312734482e-06]), ([0.0, 0.02, 0.14, 0.15, 0.28, 0.7, 1.0], [1.0, 0.5868144023703821, 0.42801264796339916, 0.06802703408885441, 0.026418567510431108, 0.005603758469325122, 0.005603758469325122]), ([0.0, 0.02, 0.04, 0.07, 0.16, 0.29, 0.58, 1.0], [1.0, 0.7952447955464811, 0.5438803839776275, 0.25972095871455547, 0.08556125901857041, 0.009486261980391078, 0.013330856811433292, 0.013330856811433292]), ([0.0, 0.03, 0.09, 0.12, 0.13, 0.17, 0.24, 0.27, 0.28, 1.0], [1.0, 0.38311239376886613, 0.4895999992938843, 0.17533993897181818, 0.19871829763325555, 0.30404239884903067, 0.22925730237669129, 0.03244319516684969, 0.0015726890147184168, 0.0015726890147184168]), ([0.0, 0.03, 0.06, 0.09, 0.15, 0.42, 1.0], [1.0, 1.4006295339682588, 0.5765733987752932, 0.6526253762447717, 0.12358220498831836, 0.0388812177896038, 0.0388812177896038]), ([0.0, 0.02, 0.03, 0.04, 0.06, 0.31, 0.66, 1.0], [1.0, 0.7871795036313491, 1.0740759251345378, 0.2714810558592753, 0.09864215564787089, 0.05309726072368255, 0.03196839837221992, 0.03196839837221992]), ([0.0, 0.02, 0.24, 0.78, 0.96, 1.0], [1.0, 0.4921442638650503, 0.10640857573956038, 0.1644356293201155, 0.0501396132774123, 0.0501396132774123]), ([0.0, 0.07, 0.1, 0.74, 1.0], [1.0, 0.33039533424704004, 0.07244432428306204, 0.10406716746889602, 0.10406716746889602]), ([0.0, 0.02, 0.04, 1.0], [1.0, 0.294857684409996, 0.08269200670748178, 0.08269200670748178]), ([0.0, 0.02, 0.44, 0.96, 0.97, 1.0], [1.0, 0.03543063081356857, 0.004896322238676758, 0.10320348648797963, 0.13406425809166603, 0.13406425809166603]), ([0.0, 0.02, 0.22, 0.25, 0.29, 1.0], [1.0, 0.1629698294263568, 0.3554392408614004, 0.025695764205155357, 0.054003361889052026, 0.054003361889052026]), ([0.0, 0.05, 0.1, 0.26, 0.56, 1.0], [1.0, 0.4246314011479331, 0.1540592805512325, 0.08549454817525015, 0.09304677757437718, 0.09304677757437718]), ([0.0, 0.02, 0.09, 0.27, 0.87, 1.0], [1.0, 0.3749971332599069, 0.0523930179542975, 0.03319552429601982, 0.08969629857358392, 0.08969629857358392]), ([0.0, 0.08, 0.18, 0.29, 1.0], [1.0, 0.549233445859992, 0.04075211429204547, 0.047409810789472547, 0.047409810789472547]), ([0.0, 0.06, 0.55, 1.0], [1.0, 0.11692317768946442, 0.08462751699090552, 0.08462751699090552]), ([0.0, 0.03, 0.7, 1.0], [1.0, 0.13727899492980367, 0.041823771099217574, 0.041823771099217574]), ([0.0, 0.02, 0.06, 0.2, 0.4, 1.0], [1.0, 0.4037267722470325, 0.2042422005424591, 0.07438123532267371, -0.004000860632170666, -0.004000860632170666]), ([0.0, 0.03, 0.04, 1.0], [1.0, 0.27588108358335756, 0.05233915638121652, 0.051741966766216584]), ([0.0, 0.04, 0.43, 0.54, 1.0], [1.0, 0.11305882347541406, 0.15376660901496383, 0.028090085997522572, 0.028090085997522572]), ([0.0, 0.02, 0.07, 0.13, 0.37, 1.0], [1.0, 0.8189265848067194, 0.5280510552940025, 0.043861502857534665, -0.015256806691714624, -0.015256806691714624]), ([0.0, 0.14, 0.16, 0.2, 0.32, 0.6, 1.0], [1.0, 0.9148323300911172, 0.8815918815789007, 0.17293662646663394, 0.04553751917382274, 0.06134993325365781, 0.06134993325365781]), ([0.0, 0.03, 0.13, 0.15, 0.39, 0.72, 1.0], [1.0, 0.15069627900050872, 0.21924689942977404, 0.11036772443923935, 0.0189411720138519, 0.021320050068764, 0.021320050068764])]"

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
