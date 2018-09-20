import PySimpleGUI as sg
import label_chord as lb
import label_wav
import time

labels_list = "/Users/dave/projects/PyPitch/src/chord_commands_train/conv_labels.txt"
path_to_graph= "/Users/dave/projects/PyPitch/src/my_graph4.pb"
wav_file_path = "/Users/dave/tensorflow/tensorflow/examples/speech_commands/temp.wav"

chord_map = {'cmaj':'C-Major(C E G)', 'dmaj':'D-Major (D F# A)', 'emaj':'E-Major(E G# B)',
             'fmaj':'F-Major (F A C)','gmaj':'G-Major(G B D)', 'amaj':'A-Major(A C# E)', 'bmaj':'B-Major(B D# F#)',
             'cmaj7':'C-Major7(C E G B)', 'dmaj7':'D-Major7(D F# A C#)','emaj7': 'E-Major7(E G# B D#)',
             'fmaj7':'F-Major7(F A C E)', 'gmaj7':'G-Major7(G B D F#)', 'amaj7': 'A-Major7(A C# E G3)',
             'bmaj7':'B-Major7(B D# F# A#', 'cmin':'C-Minor(C Eb G)', 'dmin':'D-Minor(D F A)', 'emin':'E-Minor(E G B)',
             'fmin':'F(F Ab C)' ,'gmin':'G-Minor(G Bb D)', 'amin':'A-minor(A C E)','bmin':'B-Minor(B D F#)'}

def listenButton(txt):
    newText = sg.Text('Listening',size = (10,4))
    txt.Update(newText)

def PyPitchGUI():
    sg.SetOptions(text_justification='center')
    form = sg.FlexForm('PyPitch', font=("Helvetica"),)
    text1 = sg.Txt('Press Listen then play a chord', size = (25,4))
    text2 = sg.Text('The Chord is: ', size = (30,4))
    
    layout =[
        [sg.Text('PyPitch',size = (18,4), font=('Helvetica', 25))],
        [sg.T(' ' * 5), sg.ReadFormButton('Listen'),sg.Quit()],
        [text1],
        [text2],
    ]
    form.Layout(layout)
    #del(form)
    sg.SetOptions(text_justification='left')

    while True:
        button,value = form.Read()
        if button == 'Listen':
            chord = lb.listen(path_to_graph, labels_list, wav_file_path)
                     
            text2.Update(chord_map[chord])
            text1.Update('Waiting')
        
        elif button == 'Quit' or button is None:
            break

if __name__ == '__main__':
    PyPitchGUI()
