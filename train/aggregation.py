import torch

def aggregate_lora():
    clients_path = ['/data/qjx/logad/client_1/epoch_1.pth', 
                    '/data/qjx/logad/client_2/epoch_1.pth/',
                    '/data/qjx/logad/client_3/epoch_1.pth/']
    
    model_agg = torch.load(clients_path[0])
    for model_path in clients_path[1:]:
        model = torch.load(model_path)
        for k in model['state_dict'].keys():
            model_agg['state_dict'][k] += model['state_dict'][k]
        
    model_agg['state_dict'][k] /= len(clients_path)  # average weights
    
    # with open('/data/qjx/logad/aggregation/agg_1.pth', 'w') as f:
    torch.save(model_agg, '/data/qjx/logad/aggregation/agg_1.pth')
    
if __name__ == '__main__':
    # aggregate_lora()
    torch.load('/data/qjx/logad/client_2/epoch_0.pth/', )