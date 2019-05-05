import Random

#=
using PyCall
const chainer = pyimport("chainer")
const math = pyimport("math")
const np = pyimport("numpy")
const Chain = chainer.Chain
const F = chainer.functions
const L = chainer.links
const SerialIterator = chainer.iterators.SerialIterator
const training = chainer.training
const extensions = training.extensions
const optimizers = chainer.optimizers
=#

function get_mnist()
    mnist = chainer.datasets.mnist
    train, test = mnist.get_mnist()
end


function get_model()
    basemodel = @pydef mutable struct MNIST <: Chain
        function __init__(self, n_hidden=100)
            pybuiltin(:super)(MNIST, self).__init__()
            @pywith self.init_scope() begin
                self.l1 = L.Linear(nothing, n_hidden)
                self.l2 = L.Linear(nothing, n_hidden)
                self.l3 = L.Linear(nothing, 10)
            end
        end

            function __call__(self, x)
                h = F.relu(self.l1(x))
                h = F.relu(self.l2(h))
                h = self.l3(h)
                return h
            end
    end

    L.Classifier(basemodel())
end


function train()
    batchsize = 128
    epochs = 10
    train_set, _ = get_mnist()
    N=pybuiltin(:len)(train_set)
    train_set, val_set = chainer.datasets.split_dataset(
            train_set,
            convert(Int, 0.9*N),
            Random.shuffle(collect(0:N-1))
        )
    train_iter = SerialIterator(train_set, batchsize)
    test_iter = SerialIterator(val_set, batchsize, repeat=false, shuffle=false)
    model = get_model()
    optimizer = optimizers.MomentumSGD()
    optimizer.setup(model)
    updater = training.updaters.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epochs, "epoch"), out="result")
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.PrintReport(
        ["epoch", "main/loss", "main/accuracy",
         "validation/main/loss", "validation/main/accuracy", "elapsed_time"])
    )
    trainer.extend(extensions.snapshot_object(model, filename="bestmodel.npz"),
                   trigger=training.triggers.MinValueTrigger("validation/main/loss"))
    trainer.run()
end


function predict()
    model = get_model()
    _, test_set = get_mnist()
    chainer.serializers.load_npz("result/bestmodel.npz", model)

    counter = 0

    @pywith chainer.using_config("train", false) begin
        @pywith chainer.function.no_backprop_mode() begin
            for t in test_set
                img, label = t
                y = model.predictor(np.expand_dims(img,axis=0))
                predict = np.argmax(np.squeeze(y.array))
                if predict == label
                    counter += 1
                end
            end
        end
    end

    acc = counter/pybuiltin(:len)(test_set)
    println("accuracy for test set = $(100*acc) [%]", )
end

