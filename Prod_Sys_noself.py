# import numpy as np
# import random

# routing_layout = [[1, 2],
#                   [1, 3],
#                   [2, 4],
#                   [2, 5],
#                   [3, 5],
#                   [3, 6]]
# machine_layout = [1, 2, 2, 2, 1, 1]
# num_products = 3
# Interarrival_Time_mean = 30.0
# Interarrival_Time_StDev = 5.0
# num_machines = 2
# num_edges = 6
# node_product = [200, 0, 0, 0, 0, 0]
# machine_layout = [1, 2, 2, 2, 1, 1]
# processing_time_mean =  [4, 3, 5, 4, 4, 3]
# processing_time_StDev = [1, 1, 2, 1, 1, 1]
# product_batch_prob = [0.5, 0.35, 0.15]
# time_horizon = 600
# batch = 10
# n_sets = 200
# rng_list = [[[4, 1], 0, 0, 0, [4, 1], [3, 1]], [0, [3, 1], [5, 2], [4, 1], 0, 0], [2, 3, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 3, 3, 1, 2, 1, 1], [34.61411563302716, 75.44956720231443, 108.19790490441284, 141.09566260486486, 169.14425833768883, 196.2346301659551, 218.44228862209502, 243.31745795775439, 274.52329679191325, 318.46679602608606, 356.08251280318586, 386.0633798448415, 417.04327524660596, 449.96384808493997, 481.0016885868974, 507.34078739543963, 542.894610504102, 563.0014618891969, 595.8482941821354]]


# def previous_node(num_nodes, node, possible_node):    # Returns list of predecesors
#     for i in range(num_nodes):       
#         if node == routing_layout[i][1]:
#             pre_node = routing_layout[i][0]
#             possible_node.append(pre_node)
#     if ((len(possible_node)>0) and (not(1 in possible_node))):
#         for element in possible_node:
#             if pre_node == 1:
#                 break
#             previous_node(num_nodes, element, possible_node)                   
#     return(possible_node)
           
# def check_node(node_product, end_nodes, product):                                               # Return inventory and corresponding node  
#     node = end_nodes[product-1]                                                                 # Product's end node from list                                                
#     inventory = node_product[node-1]                                                            # Inventory at node from replicated list of intermediate product
#     if inventory != 0:                                                                          # Updates inventory at end-node
#         possible_node = node
#     else:
#         possible_node = []
#         previous_node(num_nodes, node, possible_node)
#         length= len(possible_node)
#         for i in reversed(range(length)):
#             inventory = node_product[possible_node[i-1]-1]
#             if inventory == 0:
#                 possible_node.remove(possible_node[i-1])
#     return inventory, possible_node
#         # '''
#         # def get_sequence(product):           # Returns possible routing sequences with inventory
#         #     end = end_nodes[product-1]
#         #     possible_seq = []
#         #     nodes = previous_node(num_nodes, end, possible_seq)
#         #     nodes.reverse()
#         #     nodes.append(end)                                               # List of predecesors
#         #     in_route = check_node(node_product, end_nodes, rng_list[2][0])  # List of nodes with inventory
#         #     invent_route = []                                               # Empty list for predecessors with inventory
#         #     routes = []                                                 # Empty list for routes of inventory and end node 
#         #     for i in in_route:
#         #         for j in nodes:
#         #             if i == j:
#         #                 invent_route.append(j)
#         #             invent_route.append(end)
#         #         routes.append(invent_route)
#         #     return routes
#         #     print("Routing sequence for product ", product, ": ", nodes)
#         #     if len(nodes)>num_products:
#         #         seq = np.arange(len(nodes)/(num_products-1))
#         # '''
# def get_sequence(product):
#     end = end_nodes[product-1]
#     possible_seq = []
#     nodes = previous_node(num_nodes, end, possible_seq)
#     nodes.reverse()
#     nodes.append(end)
#     if len(nodes)>1+num_products:
#         seq = np.arange(len(nodes)/(num_products-1))
#         nodes = [[1, 2, 5], [1, 3, 5]]
#     return nodes

# def get_sequence_time(seq):
#     edges = []
#     for i in range(len(routing_layout)):
#         for j in range(len(seq)):
#             if routing_layout[i][0] == seq[j] and routing_layout[i][1] == seq[j+1]:
#                 edges.append(i)
#     total_time = 0
#     order_time = []
#     for i in edges:
#         time = random.normalvariate(processing_time_mean[i], processing_time_StDev[i])
#         #print("random time ", time)
#         order_time.append(time)
#         sum_order_time = sum(order_time)
#     print(order_time, sum_order_time)
#     return(edges, order_time)

# def update_time(prod):
#     min_time = time_horizon
#     invent, invent_seq = check_node(node_product, end_nodes, prod)
#     seq = get_sequence(prod)
#     if type(seq[0]) == list:
#         min_time = time_horizon
#         for i in range(len(seq)):
#             edges, time = get_sequence_time(seq[i])
#             if sum(time) < min_time:
#                 min_time = sum(time)
#                 optimal_time = time
#                 optimal_edges = edges
#     else:
#         optimal_edges, optimal_time = get_sequence_time(seq)
#     machines = []
#     #print("optimal edges: ", optimal_edges)
#     for elem in optimal_edges:
#         machines.append(machine_layout[elem])
#     t = 0
#     print("time: ", optimal_time)
#     for i in range(len(machines)):
#         t += optimal_time[i]
#         for j in range(len(machine_layout)):
#             if machine_layout[j] == machines[i]:

#                 edge_time[j] = t + clock
#     print("machines ", machines)
#     print("Edges time: ", edge_time)
#     network_time.append(sum(optimal_time))
#     print("Net work time: ", network_time)

# # MAIN CODE

# # LIST RANDOM NUMBERS GENERATED
# # for j in range(num_machines):                   # Generate/attach random machine processing times for # of machines
# #     list_initiator = []                          
# #     for i in range(num_edges):
# #         if machine_layout[i] == j+1:
# #             parameters = [processing_time_mean[i], processing_time_StDev[i]]
# #             list_initiator.append(parameters)
# #         else:
# #             list_initiator.append(0)
# #     rng_list[j] = list_initiator
# # product_orders_rng = []
# # arrival_times_rng = []

# # orders_time = 0   
# # num_orders = 0
# # for i in range(time_horizon):                        # Generate random order inter-arrival times
# #     order_arrival_time = random.normalvariate(Interarrival_Time_mean, Interarrival_Time_StDev)
# #     orders_time += order_arrival_time                                                           # Sum of arrival times                                                                                                                                                                 
# #     if orders_time <= time_horizon:                                                             # Attach if sum is less than time horizon
# #         arrival_times_rng.append(orders_time)                                                   # Track number of orders
# #         num_orders += 1
# #         product = random.choices(np.arange(1, num_products+1), weights = product_batch_prob, k = 1)
# #         product_orders_rng.append(product[0])
# #     else: 
# #         break
# # rng_list[-2] = product_orders_rng
# # rng_list[-1] = arrival_times_rng

# print(rng_list)
# print("")
# # CREATING END NODE LIST
# num_nodes = routing_layout[num_edges-1][1]
# end_nodes = []
# for i in range(num_products): (end_nodes.append(num_nodes-i))
# print(end_nodes)
# end_nodes.reverse()
# network_time = []
# edge_time = np.zeros(len(machine_layout))

# leadtime = []
# clock = rng_list[3][0]
# i = 0
# tot_time = 0
# product = rng_list[2][0]
# print("Product: ", product)
# update_time(product)
# print("")
# print("edges time: ",edge_time)
# while len(leadtime) <= len(rng_list[2]):
#     i += 1
#     print("Clock: ", clock)
#     print(max(edge_time))
#     print(rng_list[3][i])
    
#     if max(edge_time) > rng_list[3][i]:
#         clock = max(edge_time)
#         print("here")
#     else:
#         clock = rng_list[3][i]
#         product = rng_list[2][i]
#         print("Product: ", product)
#         update_time(product)
#         print("")
#         print("edges time: ",edge_time)
#         print("")
#         print("CL", clock)
#     if i == 5:
#         break


# print(network_time)
# num_edges = len(routing_layout)
# rng_list = [0, 0, 0, 0]    # FIX THIS
# product = 1

# def get_lead_time(end_nodes, product, rng_list):
#     print("")

# print("")
# print("")
# x = [[1, 2],
#                             [1, 3],
#                             [2, 4],
#                             [2, 5],
#                             [3, 5],
#                             [3, 6]]
# print(len(x))

# import numpy as np

# machines = [2,1]
# empty_list = []
# machines_qeue = [1,2]

# # machines_qeue = [[]] * 2
# print(machines_qeue)
# machines_qeue = [[],[]]
# print(machines_qeue)

# optimal_time = [3.1,4.2]
# for i in machines:
#     print(machines_qeue[i-1], i-1)
#     print(optimal_time[i-1])
#     machines_qeue[i-1] += optimal_time[i-1]
#     print("Machine Queu:", machines_qeue)
# next_inqeue = min(machines_qeue)
# ind = machines_qeue.index(next_inqeue)

# print(ind)

# seq = [[0, 3], [1, 4]]
# optimal_edges = []
# machines_qeue = [33.233644869332984, 36.448758534336314]
# edges = []
# for i in range(len(seq)):
#     time = []
#     for j in seq[i]:
#         time.append(self.factors["processing_time_mean"][j])
#     if sum(time) < min_time:
#         min_time = sum(time)
#         optimal_edges = seq[i]

routing_layout = [[1, 2],
                [1, 3],
                [2, 4],
                [2, 5],
                [3, 5],
                [3, 6]]
num_nodes = 6
end_nodes = [4,5,6]
num_products = 3
num_edges = 6
node_product = [200,100,100,0,0,0]

def previous_node(node, possible_node, check):    # Returns pre node
    pre_node = 0
    i = False
    j = num_edges
    t = 0
    if check == 0:
        while i == False:
            if node == routing_layout[j-1][1]:
                pre_node = routing_layout[j-1][0]
                i = True
            j -= 1  
    else:
        t = 1
        while t == check:
            if node == routing_layout[j-t][1]:
                t += 1
                pre_node = routing_layout[j-t][0]
            j -= 1  
    return(pre_node)

def check_node(product):                                               # Return inventory and corresponding node    
    i = False 
    node = end_nodes[product-1]
    possible_node = []                                                                 # Product's end node from list                                                
    k =0                                       # Inventory at node from replicated list of intermediate product
    if product != 1 and product != num_products:
        check = 0
        for j in range(num_nodes):
            if routing_layout[j][1] == node:
                check += 1
        
        for j in range(check):
            node = end_nodes[product-1]                                                                 # Product's end node from list                                                
            inventory = node_product[node-1]  
            lst_nodes = [node]
            while inventory == 0 or i == False:
                if previous_node(node,possible_node,0) == 1 and j != 0:
                    if node_product[node-1] == 0:
                        lst_nodes.append(1)
                        break
                    else:
                        break
                print("haha",node)
                node = previous_node(node, lst_nodes,j)
                inventory = node_product[node-1]
                if inventory != 0:
                    i = True
                lst_nodes.append(node)
                if k == 5: break
                k+=1
                print(node)

            possible_node.append(lst_nodes)
    else:
        inventory = node_product[node-1] 
        print(inventory) 
        possible_node = [node]
        while inventory == 0 and i == False:
            node = previous_node(node, possible_node,0)
            inventory = node_product[node-1]
            if inventory != 0:
                i = True
            possible_node.append(node)
        
    print(possible_node)




possible_node =[]
check_node(product = 2)

