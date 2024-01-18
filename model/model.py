import torch
from helper.image_transform import process_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0) 
    image = image.to(device)
    
    # to make sure no drop; otherwise, the output is random
    model.to(device)
    model.eval()
    
    with torch.no_grad ():
        output = model.forward(image)
        
    output_prob = torch.exp(output)
    
    probs, indeces = output_prob.topk(topk)
    probs   =   probs.to('cpu').numpy().tolist()[0]
    indeces = indeces.to('cpu').numpy().tolist()[0]
    
    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[item] for item in indeces]
    
    return probs, classes