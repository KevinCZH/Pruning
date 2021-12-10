import matplotlib.pyplot as plt
photopath = os.path.join('photos')
if not os.path.exists(photopath):
    os.makedirs(photopath)
conv_id = 0
bn_id   = 0
for m0 in model.modules():
	if isinstance(m0, nn.Conv2d):
		counts, bins = np.histogram(m0.weight.data.view(-1).cpu().numpy())
		plt.hist(bins[:-1], bins, weights=counts)
		plt.xlabel('Weight')
		# save
		# 適用於儲存任何 matplotlib 畫出的影象，相當於一個 screencapture
		fig = os.path.join(photopath, 'conv'+ str(conv_id) + '.png')
		plt.savefig(fig)
		plt.cla()
		conv_id += 1
	elif isinstance(m0, nn.BatchNorm2d):
		counts, bins = np.histogram(m0.weight.data.view(-1).cpu().numpy())
		plt.hist(bins[:-1], bins, range = (0,1), weights=counts)
		plt.xlabel('gamma')
		# save
		# 適用於儲存任何 matplotlib 畫出的影象，相當於一個 screencapture
		fig = os.path.join(photopath, 'bn'+ str(bn_id) + '.png')
		plt.savefig(fig)
		plt.cla()
		bn_id += 1