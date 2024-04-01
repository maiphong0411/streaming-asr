import wenet
import time
import pandas as pd
# model = wenet.load_model('chinese')
model = wenet.load_model(model_dir='/vinbrain/phongmt/wenet/wenet/examples/foo/s0/exp/sp_spec_aug/', gpu=2)
test_ds = pd.read_csv("/vinbrain/phongmt/wenet/wenet/examples/foo/s0/data/test/wav.scp", header=None, delimiter=" ")
# print(test_ds.head())
# print(test_ds[0][1])
import time
chunk_size_list = [8]
for chunk_size in chunk_size_list:
    start = time.time()
    with open(f"text", 'a') as f:
        for i in range(len(test_ds)):
            print(i)
            print(test_ds.iloc[i][1])
            result = model.transcribe(test_ds.iloc[i][1], simulate_stream=True, chunk_size=chunk_size)
            print(result)
            predicted_text = result['output']
            formated_number = '{:04d}'.format(test_ds.iloc[i][0])
            f.write(f"{formated_number} {predicted_text}\n")
            print('\n======================\n')
    end = time.time()
    print("logging time ", end - start)