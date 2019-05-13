def dataloader(source):
    
     data_dir = source
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225] )])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                         ])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(), 
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225] )
                                         ])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform= train_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform = testing_transforms)    
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size =  64, shuffle = True)
    validtnloader = torch.utils.data.DataLoader(validation_dataset, batch_size =  32)
    testloader = torch.utils.data.DataLoader(testing_dataset, batch_size =  32)
    return trainloader, validatnloader, testloader

def tom_network(hidden_size):
    if structure == 'vgg13':
        model = models.vgg13(pretrained = True)
    elif structure == 'densenet121' :
        model = models.densenet121(pretrained = True):
    else:
        print("Choose either vgg13 and densent121")
    input_size = 25088
    output = 102
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(input_size, hidden_size)),
                                       ('relu', nn.ReLU()),
                                       ('dropout', nn.Dropout(0.05)),
                                       ('fc2',  nn.Linear(hidden_size,output_size)),
                                       ('output', nn.LogSoftmax(dim=1))]))
                     


    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    return model, criterion, optimizer


def train_model(model,trainloader, validatnloader, epochs, print_every, criterion, optimizer ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    steps=0
    
    for e in range(epochs):
        running_loss = 0

        for images, labels in iter(trainloader):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output,labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()


            if steps % print_every == 0:
                validtn_loss = 0
                accuracy = 0

                model.eval()
                with torch.no_grad():
                    for images, labels in iter(validtnloader):

                        images, labels = images.to(device), labels.to(device)

                        validtn_output = model.forward(images)
                        validtn_loss += criterion(validtn_output, labels).item()

                        ps = torch.exp(validtn_output)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()


                        #top_p, top_class = ps.topk(1, dim=1)
                        #equals = top_class == labels.view(*top_class.shape)

                        #accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                #print(labels)
                print("Epoch: {}/{}".format(e+1,epochs), 
                            "train_loss: {:.3f}".format(running_loss/print_every),
                             "validtn_loss: {:.3f}".format(validtn_loss/len(validtnloader)),
                              "validtn_accuracy: {:.3f}".format(accuracy/len(validtnloader)))

                running_loss = 0
                model.train()

def save_checkpoint(classifier, epochs,model, optimizer ):
    
    checkpoint = {"epoch": epochs,
                  "classifier": model.classifier,
                 "optimizer": optimizer.state_dict(),
                 "state_dict": model.state_dict(),
                 "class_to_idx": train_dataset.class_to_idx}

    torch.save(checkpoint, "checkpoint.pth")
    return checkpoint

def load_chkpt(model, optimizer, filepth):
    checkpoint = torch.load(filepth)
    model.load_state_dict(checkpoint["state_dict"])
    model.classifier=checkpoint['classifier']
    optimizer.load_state_dict(checkpoint["optimizer"])
    model.class_to_idx =checkpoint['class_to_idx']
    start_epoch = checkpoint["epoch"]
    
    return model, optimizer, start_epoch

def process_image(image_path = "/home/workspace/aipnd-project/flowers/test/1/image_06752.jpg"):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    #pil_image = Image.open(f'{image}' + '.jpg')
    pil_image = Image.open(image_path)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize])
    
    
    np_image = np.array(preprocess(pil_image))
    return np_image

def predict(image_path= "/home/workspace/aipnd-project/flowers/test/1/image_06752.jpg", model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # TODO: Implement the code to predict the class from an image file
    
    model.eval()
    
    #loading the model from the checkpoint
    model_loaded = load_chkpt(model, optimizer, "checkpoint.pth")[0].to(device)
    
    
    img = process_image(image_path)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    image = img_tensor.unsqueeze_(0)
    
    image = image.to(device)
    
    with torch.no_grad():
        output = model_loaded.forward(image)
        
    ps = torch.exp(output)
    probs, classes =ps.topk(topk)
    
    
    probs_top = np.array(probs)[0]
    classes_top = np.array(classes[0])
    
    class_to_idx = model_loaded.class_to_idx
    inv_class_to_idx = {v: k for k, v in model_loaded.class_to_idx.items()}
    
    top_classes = [inv_class_to_idx[x] for x in classes_top]
    return probs_top, top_classes