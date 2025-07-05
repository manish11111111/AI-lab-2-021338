from itertools import product

def generate_date(n):
    data = []
    for inputs in product([0, 1], repeat=n):
        target = 1 if all(inputs) else 0
        data.append((list(inputs), target))
    return data

def print_table(data, n):
    print(f"Truth Table with {n} inputs for AND gate:")
    labels = [chr(65 + i) for i in range(n)]
    print("".join(labels) + "\tOutput(Y)")
    for row in data:
        inputs = "".join(map(str, row[0]))
        output = row[1]
        print(f"{inputs}\t\t{output}")

def print_output(data, weights, b, n):
    print("\nTruth Table with Learned Outputs (after training):")
    for row in data:
        inputs = row[0]
        target = row[1]
        s = sum(weights[i] * inputs[i] for i in range(n)) + b
        out = 1 if s >= 0 else 0
        inputs_str = "".join(map(str, inputs))
        print(f"{inputs_str}\t{out}")


def main():
    print("Enter the no of inputs for AND Gate:")
    n = int(input())
    data = generate_date(n)
    print_table(data, n)

    weights = [0.0] * n
    lf = 0.1
    b = 0

    epoch = 0
    max_epochs = 100
    converged = False

    while epoch < max_epochs and not converged:
        epoch += 1
        total_errors = 0

        for ins, target in data:
            s = sum(weights[i] * ins[i] for i in range(n)) + b
            out = 1 if s >= 0 else 0

            if target != out:
                total_errors += 1

                for i in range(n):
                    weights[i] += lf * (target - out) * ins[i]
                b += lf * (target - out)

        print(f"Epoch {epoch}: Total Errors = {total_errors}")

        if total_errors == 0:
            converged = True
            print("Training Converged!")
            print("Final Weights:", weights)
            print("Final Bias:", b)
            print_output(data, weights, b, n)

    if not converged:
        print("Training did not converge within the maximum number of epochs.")
        print("Final Weights:", weights)
        print("Final Bias:", b)

if __name__ == "__main__":
    main()
