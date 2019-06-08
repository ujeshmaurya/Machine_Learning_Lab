	def forward(self, X):
		self.a = []
		self.z = []
		self.delta = []
		self.a.append(X)
		for layer in range(self.number_of_layers):
			if self.bias:
				self.a[layer] = np.append(self.a[layer], np.ones((1, 1)), axis=0)
			self.z.append(dot(self.W[layer], self.a[layer]))
			# print(activate(self.z[layer], self.activation_func[layer]).shape)
			self.a.append(activate(self.z[layer], self.activation_func[layer]))
		return self.a[-1]
		
	def backward(self, gradient):
		self.delta = [np.array([]) for layer in range(self.number_of_layers + 1)]
		if self.bias:
			self.delta[self.number_of_layers] = np.append(gradient, np.ones((1, 1)), axis=0)
			for layer in range(self.number_of_layers - 1, -1, -1):
				self.delta[layer] = dot(np.transpose(self.W[layer]), self.delta[layer + 1][:-1]) * self.a[layer] * (1 - self.a[layer])
			for layer in range(self.number_of_layers):
				self.W[layer] = self.W[layer] + self.alpha * dot(self.delta[layer + 1][:-1], np.transpose(self.a[layer]))
		else:
			self.delta[self.number_of_layers] = gradient
			for layer in range(self.number_of_layers - 1, -1, -1):
				self.delta[layer] = dot(np.transpose(self.W[layer]), self.delta[layer + 1][:-1]) * self.a[layer] * (1 - self.a[layer])
			for layer in range(self.number_of_layers):
				self.W[layer] = self.W[layer] + self.alpha * dot(self.delta[layer + 1], np.transpose(self.a[layer]))