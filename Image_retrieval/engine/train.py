


def train(loss_fun, optimizer, model, train_loader, center, optimizer_center, cetner_loss_weight):
    model.train()
    for step, (feature, label) in enumerate(train_loader):
        feature = feature.cuda()
        label = label.cuda()
        feat, target = model(feature)
        feat = feat.cuda()
        target = target.cuda()
        loss = loss_fun(feat, target, label, center).cuda()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        loss.backward()
        optimizer.step()
        for param in center.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()
    return loss


