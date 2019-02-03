
def basic_model_metrics(model):
    std = []
    for param in model.parameters(): 
        std.append(param.data.std().item())
       
    return np.array(std)