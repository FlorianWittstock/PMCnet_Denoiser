### OLD VERSINS
#old version
def old_calculate_grad_regression_large(Weight_0,B_0,tp):
    L = tp['L']
    X = tp['x_0'].cpu().double()
    y = tp['y'].cpu()
    prior_W = tp['prior_W']
    prior_b = tp['prior_b']
    regularization_weight = tp['regularization_weight']
    activation = tp['activation']
    classification = tp['classification']

    W1 = Variable(Weight_0[1], requires_grad=True)
    b1 = Variable(B_0[1], requires_grad=True)
    W2 = Variable(Weight_0[2], requires_grad=True)
    b2 = Variable(B_0[2], requires_grad=True)
    W3 = Variable(Weight_0[3], requires_grad=True)
    b3 = Variable(B_0[3], requires_grad=True)
    
    parameters = [W1, b1, W2, b2, W3, b3]
    
    optimizer = optim.Adam(parameters)
    
    W = {}
    b = {}
    
    W[1] = W1
    W[2] = W2
    W[3] = W3
    b[1] = b1
    b[2] = b2
    b[3] = b3
    
    output = FFnetwork(W,b,L,X,activation)
            
    if classification == 'binary':        
        y = y.float()
        loss_ = nn.BCEWithLogitsLoss(reduction='sum')
        loss = loss_(output, y)
    elif classification == 'regression':
        y = y.double()
        loss_ = nn.MSELoss(reduction='mean')
        loss = loss_(output, y)
    else:
        loss = F.cross_entropy(output, y, reduction='sum')# the MSE part
        
    l1_penalty = torch.tensor(0.0)  
    l2_penalty = torch.tensor(0.0)  
    if prior_W == 'L1':  
        l1_penalty = regularization_weight * sum([p.abs().sum() for p in parameters])
    elif prior_W == 'L2':    
        l2_penalty = regularization_weight * sum([(p**2).sum() for p in parameters])
    loss_with_penalty = loss + l1_penalty + l2_penalty
    
    
    optimizer.zero_grad()
    loss_with_penalty.backward()
    optimizer.step()
    
    W_g = {}
    B_g = {}
    
    W_g1 = optimizer.param_groups[0]['params'][0].grad
    W_g[0] = W_g1
    W_g2 = optimizer.param_groups[0]['params'][2].grad
    W_g[1] = W_g2
    W_g3 = optimizer.param_groups[0]['params'][4].grad
    W_g[2] = W_g3
    B_g1 = optimizer.param_groups[0]['params'][1].grad
    B_g[0] = B_g1
    B_g2 = optimizer.param_groups[0]['params'][3].grad
    B_g[1] = B_g2
    B_g3 = optimizer.param_groups[0]['params'][5].grad
    B_g[2] = B_g3
    
    return W_g,B_g,loss_with_penalty

def old_Vec2param(vector,tp):
    # save W and b as dictionary
    L = tp['L']
    M = tp['M']
    W = {}
    b = {}
    for ll in range(L):
        W[ll+1] = torch.transpose(vector[:M[ll]*M[ll+1]].reshape(M[ll+1],M[ll]),0,1)
        W[ll+1] = Variable(W[ll+1], requires_grad=True) 
        vector = vector[M[ll]*M[ll+1]:]
        b[ll+1] = vector[:M[ll+1]].reshape(M[ll+1])
        b[ll+1] = Variable(b[ll+1], requires_grad=True)
        vector = vector[M[ll+1]:]
    return W, b 

def old_Param2vec(W,b,tp):
    # combine W and b to a vector
    L = tp['L']
    M = tp['M']
    dimension = tp['dimension']
    vector = torch.zeros(dimension,1).cpu()
    ind = 0
    for ll in range(L):
        vector[ind:ind+M[ll]*M[ll+1],:]=torch.transpose(W[ll],0,1).reshape(M[ll]*M[ll+1],1)
        ind = M[ll]*M[ll+1]
        vector[ind:ind+M[ll+1],:]=b[ll].reshape(M[ll+1],1)
        ind = ind+M[ll+1]
    return vector