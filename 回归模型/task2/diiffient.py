f = open(r'test/test_labels1.txt', 'r')
labels_huigui = f.read().splitlines(False)
f.close()
f = open(r'test/test_labels_beiyesi.txt', 'r')
labels_beiyesi = f.read().splitlines(False)
f.close()
print(len(labels_huigui), len(labels_beiyesi))
l = len(labels_beiyesi)
difcount = 0
for i in range(l):
    if labels_huigui[i] != labels_beiyesi[i]:
        difcount += 1
print('相差率为', difcount/l)

