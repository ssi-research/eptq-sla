def model_register_hook(in_net, list2append, hook_handles, layers):
    def get_activation(in_name):
        def hook(model, input, output):
            list2append.update({in_name: output})

        return hook

    for i, (name, module) in enumerate(in_net.named_modules()):
        if isinstance(module, layers):
            hook_handles.append(module.register_forward_hook(get_activation(module.name)))