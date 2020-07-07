import midi
import os
import tensorflow as tf
import numpy as np

#This file is heavily based on Daniel Johnson's midi manipulation code in https://github.com/hexahedria/biaxial-rnn-music-composition
#lower bound and upper bound restrict the pitch of selected notes
lower_bound = 24
upper_bound = 102
span = upper_bound - lower_bound

# convert midi file to notes
def midiToNote(midi_file_path,span=span):
    pattern = midi.read_midifile(midi_file_path) #a list of tracks
    time_left = []
    for track in pattern: #track is a list of midi events
        time_left.append(track[0].tick)
    posns = [0 for track in pattern]
    noteMatrix = []
    time = 0
    state = [[0,0] for x in range(span)]
    noteMatrix.append(state)
    condition = True
    while condition:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(span)]
            noteMatrix.append(state)
        for i in range(len(time_left)):
            if not condition:
                break
            while time_left[i] == 0:
                track = pattern[i]
                pos = posns[i]
                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lower_bound) or (evt.pitch >= upper_bound):
                        pass
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lower_bound] = [0, 0]
                        else:
                            state[evt.pitch-lower_bound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        out =  noteMatrix
                        condition = False
                        break
                try:
                    time_left[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    time_left[i] = None
            if time_left[i] is not None:
                time_left[i] -= 1
        if all(t is None for t in time_left):
            break
        time += 1
    S = np.array(noteMatrix)
    noteMatrix = np.hstack((S[:, :, 0], S[:, :, 1]))
    noteMatrix = np.asarray(noteMatrix).tolist()
    return noteMatrix

# read midi files, get a list of list of notes
def get_songs(midi_path):
    files = os.listdir(midi_path)
    songs = []
    for f in files:
        f = midi_path+'/'+f
        print('loading: ',f)
        try:#ensure the given song has enough note
            song = np.array(midiToNote(f))
            if np.array(song).shape[0] > 64:
                songs.append(song)
        except Exception as e:
            print('invalid data: ', e)
    print("the number of midi files that has been loaded successfully: ", len(songs))
    return songs

# the file named midi contains the midi type music(it also depends on what kinds of music we want to generate) downloaded before
songs = get_songs('midi')
note_range = upper_bound - lower_bound
# set music timestep
n_timesteps = 256
n_input = 2 * note_range * n_timesteps
n_hidden = 64

X = tf.placeholder(tf.float32, [None, n_input])
W = None  #weights
bh = None #biased matrix in hidden layer
bv = None #biased matrix in output layer

def sample(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

def gibbs_sample(k):
    def body(count, k, xk):
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))
        return count+1, k, xk
    count = tf.constant(0)

    def condition(count,  k, xk):
        return count < k
    [_, _, x_sample] = tf.while_loop(condition, body, [count, tf.constant(k), X])
    x_sample = tf.stop_gradient(x_sample)
    return x_sample

#define neural network
def neural_network():
    global W
    W  = tf.Variable(tf.random_normal([n_input, n_hidden], 0.01))
    global bh
    bh = tf.Variable(tf.zeros([1, n_hidden],  tf.float32))
    global bv
    bv = tf.Variable(tf.zeros([1, n_input],  tf.float32))

    x_sample = gibbs_sample(1)
    h = sample(tf.sigmoid(tf.matmul(X, W) + bh))
    h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

    learning_rate = tf.constant(0.005, tf.float32)
    size_bt = tf.cast(tf.shape(X)[0], tf.float32)
    W_adder  = tf.multiply(learning_rate/size_bt, tf.subtract(tf.matmul(tf.transpose(X), h), tf.matmul(tf.transpose(x_sample), h_sample)))
    bv_adder = tf.multiply(learning_rate/size_bt, tf.reduce_sum(tf.subtract(X, x_sample), 0, True))
    bh_adder = tf.multiply(learning_rate/size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))
    update = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]
    return update

# note file transfer to midi file
def noteToMidi(noteMatrix, filename="output_file", span=span):
    noteMatrix = np.array(noteMatrix)
    if not len(noteMatrix.shape) == 3:
        noteMatrix = np.dstack((noteMatrix[:, :span], noteMatrix[:, span:]))
    noteMatrix = np.asarray(noteMatrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    span = upper_bound-lower_bound
    tickscale = 55

    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    for time, state in enumerate(noteMatrix + [prevstate[:]]):
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lower_bound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lower_bound))
            lastcmdtime = time

        prevstate = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(filename), pattern)

# train neural network
def train_neural_network():
    update = neural_network()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        epochs = 2000
        batch_size = 64
        for epoch in range(epochs):
            for song in songs:
                song = np.array(song)
                song = song[:int(np.floor(song.shape[0]/n_timesteps) * n_timesteps)]
                song = np.reshape(song, [song.shape[0]//n_timesteps, song.shape[1] * n_timesteps])

                for i in range(1, len(song), batch_size):
                    train_x = song[i:i+batch_size]
                    sess.run(update, feed_dict={X: train_x})
            print(str(epoch)+' times training')

        # create model and save model
        model_dir  = "train_result"
        model_name = "midi.module"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        # save model
        saver.save(sess, os.path.join(model_dir, model_name))

        # generate midi
        sample = gibbs_sample(1).eval(session=sess, feed_dict={X: np.zeros((1, n_input))})
        S = np.reshape(sample[0,:], (n_timesteps, 2 * note_range))
        noteToMidi(S, "trained music")
        print('generated midi type file')

def main():
    train_neural_network()

if __name__ == "__main__":
    main()