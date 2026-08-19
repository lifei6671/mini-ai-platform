dataset_size = 10
batch_size = 2
epochs = 3

steps_per_epoch = dataset_size // batch_size

print("dataset_size:", dataset_size)
print("batch_size:", batch_size)
print("epochs:", epochs)
print("steps_per_epoch:", steps_per_epoch)

global_step = 0

for epoch in range(epochs):

    print(f"\n=== EPOCH {epoch + 1} ===")

    for batch_index in range(steps_per_epoch):

        global_step += 1

        print(
            f"epoch={epoch + 1}, "
            f"batch={batch_index + 1}, "
            f"global_step={global_step}"
        )