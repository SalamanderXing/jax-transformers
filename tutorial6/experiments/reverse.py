import mate
from ..data_loaders.reverse import get_loaders
from ..trainers.reverse_trainer import ReverseTrainer
from ..models.transformer import Transformer


rev_train_loader, rev_val_loader, rev_test_loader = get_loaders()

model = Transformer(
    model_dim=32,
    num_heads=1,
    num_classes=rev_train_loader.dataset.num_categories,
    num_layers=1,
    dropout_prob=0.0,
)

if mate.is_train:
    val_acc = ReverseTrainer.train_reverse(
        checkpoint_path=mate.checkpoint_path,
        seed=42,
        model=model,
        rev_train_loader=rev_train_loader,
        rev_val_loader=rev_val_loader,
        max_epochs=10,
        lr=1e-5,
        warmup=50,
    )
    mate.result({"val_acc": val_acc})
elif mate.is_test:
    ReverseTrainer.test_reverse(
        model=model,
        seed=42,
        checkpoint_path=mate.checkpoint_path,
        rev_test_loader=rev_test_loader,
        max_epochs=10,
    )
