import torch
import math
import torch.optim as optim
import torch.nn as nn
from DenoisingNetwork import ConvDenoiser

### Functions to evaluate the mixture
def diagevaluate_proposal_multiple_fullCov(x, mu, diagSig,N):
    d = x.shape[0]
    n = x.shape[1]
    K = n//N
    x=torch.transpose(x,0,1)
    mu=torch.transpose(mu,0,1)
    fp_mixt = []
    inverse_sigma, logSqrtDetSigma = diagfunctionR(diagSig,N)
    logSqrtDetSigma1 = logSqrtDetSigma.double()
    fp_mixt = torch.tensor(()).cpu()
    for k in range(K):
        x_temp = x[k*N:(k+1)*N,:]
        X0 = functionMu(x_temp, mu, N) #N*d*n
        xRinv = torch.einsum('lik,il->lik',X0, inverse_sigma.cpu()) # inverse_sigma: d*d*N
        quadform = (xRinv**2).sum(1)
        y_temp = -0.5*quadform-logSqrtDetSigma1.cpu()
        if k == 0:
            C = -y_temp.max()
        y_temp = y_temp + C
        y = (torch.exp(y_temp)).sum(0)
        fp_mixt = torch.cat((fp_mixt,y),0)
    return fp_mixt

def diagfunctionR(diagSigma,N):
    # only needs to be calculated for once
    inverse_sigma = diagSigma**(-1/2)
    logSqrtDetSigma = torch.sum(torch.log(diagSigma**(1/2)),dim=0)
    return inverse_sigma, logSqrtDetSigma

def functionMu(x, mu, N):
    s = mu.shape[0]
    n = x.shape[0]
    x_new = torch.repeat_interleave(x, repeats=s, dim=0)
    mu_new = mu.repeat(n,1)
    temp = torch.split(x_new-mu_new,s)
    temp = torch.stack(list(temp))
    output = temp.permute(1, 2, 0)
    return output


### Functions that evaluate the BNN

def evaluate_target_general_regression_large(vector,tp):
    [Weight_0,B_0] = Vec2param(vector)
    [W_g,B_g,l] = calculate_grad_regression_large(Weight_0,B_0,tp)
    return W_g,B_g,l
 

def Vec2param(vector):
    # save W and b as dictionary
    W = {}
    b = {}  
    
    # write down all the shapes
    conv1_weight_shape = (32, 1, 3, 3)
    conv1_bias_shape = (32,)  
    conv2_weight_shape = (16, 32, 3, 3)
    conv2_bias_shape = (16,)  
    conv3_weight_shape = (8, 16, 3, 3)
    conv3_bias_shape = (8,)  
    t_conv1_weight_shape = (8, 8, 3, 3)
    t_conv1_bias_shape = (8,)  
    t_conv2_weight_shape = (8, 16, 2, 2)
    t_conv2_bias_shape = (16,)  
    t_conv3_weight_shape = (16, 32, 2, 2)
    t_conv3_bias_shape = (32,)  
    conv_out_weight_shape = (1, 32, 3, 3)
    conv_out_bias_shape = (1,) 
    
    # Array containing the number of parameters for each layer
    parameters_count = [288, 32, 4608, 16, 1152, 8, 576, 8, 512, 16, 2048, 32, 288, 1]

    # Split flattened_parameters into 14 tensors
    split_parameters = torch.split(vector, parameters_count)
    
    # Create empty tensors for each layer
    conv1_weight_reconstructed = torch.empty(conv1_weight_shape)
    conv1_bias_reconstructed = torch.empty(conv1_bias_shape)
    conv2_weight_reconstructed = torch.empty(conv2_weight_shape)
    conv2_bias_reconstructed = torch.empty(conv2_bias_shape)
    conv3_weight_reconstructed = torch.empty(conv3_weight_shape)
    conv3_bias_reconstructed = torch.empty(conv3_bias_shape)
    t_conv1_weight_reconstructed = torch.empty(t_conv1_weight_shape)
    t_conv1_bias_reconstructed = torch.empty(t_conv1_bias_shape)
    t_conv2_weight_reconstructed = torch.empty(t_conv2_weight_shape)
    t_conv2_bias_reconstructed = torch.empty(t_conv2_bias_shape)
    t_conv3_weight_reconstructed = torch.empty(t_conv3_weight_shape)
    t_conv3_bias_reconstructed = torch.empty(t_conv3_bias_shape)
    conv_out_weight_reconstructed = torch.empty(conv_out_weight_shape)
    conv_out_bias_reconstructed = torch.empty(conv_out_bias_shape)
    
    # Fill them up
    conv1_weight_reconstructed = split_parameters[0].view(conv1_weight_shape)
    conv1_bias_reconstructed = split_parameters[1].view(conv1_bias_shape)
    conv2_weight_reconstructed = split_parameters[2].view(conv2_weight_shape)
    conv2_bias_reconstructed = split_parameters[3].view(conv2_bias_shape)
    conv3_weight_reconstructed = split_parameters[4].view(conv3_weight_shape)
    conv3_bias_reconstructed = split_parameters[5].view(conv3_bias_shape)
    t_conv1_weight_reconstructed = split_parameters[6].view(t_conv1_weight_shape)
    t_conv1_bias_reconstructed = split_parameters[7].view(t_conv1_bias_shape)
    t_conv2_weight_reconstructed = split_parameters[8].view(t_conv2_weight_shape)
    t_conv2_bias_reconstructed = split_parameters[9].view(t_conv2_bias_shape)
    t_conv3_weight_reconstructed = split_parameters[10].view(t_conv3_weight_shape)
    t_conv3_bias_reconstructed = split_parameters[11].view(t_conv3_bias_shape)
    conv_out_weight_reconstructed = split_parameters[12].view(conv_out_weight_shape)
    conv_out_bias_reconstructed = split_parameters[13].view(conv_out_bias_shape)
    
    # Save them in W and b
    # Weights
    W['conv1'] = conv1_weight_reconstructed
    W['conv2'] = conv2_weight_reconstructed
    W['conv3'] = conv3_weight_reconstructed
    W['t_conv1'] = t_conv1_weight_reconstructed
    W['t_conv2'] = t_conv2_weight_reconstructed
    W['t_conv3'] = t_conv3_weight_reconstructed
    W['conv_out'] = conv_out_weight_reconstructed

    # Biases
    b['conv1'] = conv1_bias_reconstructed
    b['conv2'] = conv2_bias_reconstructed
    b['conv3'] = conv3_bias_reconstructed
    b['t_conv1'] = t_conv1_bias_reconstructed
    b['t_conv2'] = t_conv2_bias_reconstructed
    b['t_conv3'] = t_conv3_bias_reconstructed
    b['conv_out'] = conv_out_bias_reconstructed

    return W, b  


# new version
def calculate_grad_regression_large(W,b,tp):

    X = tp['x_0'].cpu().double()
    y = tp['y'].cpu()
    prior_W = tp['prior_W']
    prior_b = tp['prior_b']
    regularization_weight = tp['regularization_weight']
    
    
    # Instantiate your ConvDenoiser model
    model = ConvDenoiser()

    # Load your weights and biases into the model
    model.conv1.weight.data = W['conv1']
    model.conv1.bias.data = b['conv1']
    model.conv2.weight.data = W['conv2']
    model.conv2.bias.data = b['conv2']
    model.conv3.weight.data = W['conv3']
    model.conv3.bias.data = b['conv3']
    model.t_conv1.weight.data = W['t_conv1']
    model.t_conv1.bias.data = b['t_conv1']
    model.t_conv2.weight.data = W['t_conv2']
    model.t_conv3.weight.data = W['t_conv3']
    model.conv_out.weight.data = W['conv_out']
    model.conv_out.bias.data = b['conv_out']

    # Define your optimizer
    optimizer = optim.Adam(model.parameters())
    
    # Forward pass to get model predictions
    #output = model(X)

    batch_size = 100

    # Split X into batches
    X_batches = torch.split(X, batch_size, dim=0)

    # Process each batch separately
    outputs = []
    ct = 0
    for batch in X_batches:
        ct = ct + 1
        batch_output = model(batch)
        outputs.append(batch_output)
        #print(f'Done with batch number {ct}')

    # Concatenate the results along the first dimension
    output = torch.cat(outputs, dim=0)
    
    # Convert target values to double type (assuming y is your target tensor)
    y = y.double()
    y = y.squeeze(-1)  # Remove the extra dimension with size 1

    #print(f'Output shape: {output.shape}')
    #print(f'Target shape: {y.shape}')

    # Zero out gradients to avoid accumulation from previous iterations
    optimizer.zero_grad()
    
    # Calculate the mean squared error loss between predicted and target values
    loss_ = nn.MSELoss(reduction='mean')
    loss = loss_(output, y)

    # Initialize regularization penalties
    l1_penalty = torch.tensor(0.0)
    l2_penalty = torch.tensor(0.0)

    # Check if L1 or L2 regularization is specified
    if prior_W == 'L1':
        # Calculate L1 penalty as the sum of absolute values of all model parameters
        l1_penalty = regularization_weight * sum([p.abs().sum() for p in model.parameters()])
    elif prior_W == 'L2':
        # Calculate L2 penalty as the sum of squared values of all model parameters
        l2_penalty = regularization_weight * sum([(p**2).sum() for p in model.parameters()])

    # Combine loss with regularization penalties
    loss_with_penalty = loss + l1_penalty + l2_penalty

    # Zero out gradients to avoid accumulation from previous iterations
    optimizer.zero_grad()

    # Backward pass to compute gradients of the loss with respect to model parameters
    loss_with_penalty.backward()

    # Perform optimization step to update model parameters
    optimizer.step()

    
    # Create dictionaries to store gradients
    W_g = {}
    B_g = {}

    # Iterate over all parameter names and gradients in the model
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Check if the parameter is a weight or bias
            if 'weight' in name:
                W_g[name] = param.grad.clone()
            elif 'bias' in name:
                B_g[name] = param.grad.clone()
    
    return W_g,B_g,loss_with_penalty

def Param2vec(W,b):
    
    # load from dictionaries
    conv1_weight_loaded = W['conv1.weight']
    conv2_weight_loaded = W['conv2.weight']
    conv3_weight_loaded = W['conv3.weight']
    t_conv1_weight_loaded = W['t_conv1.weight']
    t_conv2_weight_loaded = W['t_conv2.weight']
    t_conv3_weight_loaded = W['t_conv3.weight']
    conv_out_weight_loaded = W['conv_out.weight']

    conv1_bias_loaded = b['conv1.bias']
    conv2_bias_loaded = b['conv2.bias']
    conv3_bias_loaded = b['conv3.bias']
    t_conv1_bias_loaded = b['t_conv1.bias']
    t_conv2_bias_loaded = b['t_conv2.bias']
    t_conv3_bias_loaded = b['t_conv3.bias']
    conv_out_bias_loaded = b['conv_out.bias']


    # List of all parameters
    parameters = [
        conv1_weight_loaded, conv1_bias_loaded,
        conv2_weight_loaded, conv2_bias_loaded,
        conv3_weight_loaded, conv3_bias_loaded,
        t_conv1_weight_loaded, t_conv1_bias_loaded,
        t_conv2_weight_loaded, t_conv2_bias_loaded,
        t_conv3_weight_loaded, t_conv3_bias_loaded,
        conv_out_weight_loaded, conv_out_bias_loaded
    ]

    # Concatenate and reshape into a single tensor
    vector = torch.cat([param.reshape(-1) for param in parameters], dim=0).unsqueeze(1)
    return vector



def diagweightedcov(Y, w):
    w = torch.tensor(w).cpu()
    #w = w.clone().detach().cpu()
    w=w/w.sum()
    a = w @ Y  
    diagC = torch.sum(((Y-a)**2)*w.unsqueeze(-1),dim=0)
    return diagC


def crop_weights(norm_weights,fraction):
    norm_weights = norm_weights+1e-30
    K = len(norm_weights)
    d=norm_weights.tolist()
    d.sort(reverse=True)
    all_val=torch.tensor(d).cpu().double()
    #all_val = d.clone().detach()
    if fraction == 0 :
        ind = 0
    else:
        ind = math.ceil(fraction*K)
    max_val = all_val[ind]
    crop_norm_weights = torch.tensor(norm_weights).cpu().double()
    crop_norm_weights[crop_norm_weights>max_val] = max_val
    c = crop_norm_weights/torch.sum(crop_norm_weights)
    return c




