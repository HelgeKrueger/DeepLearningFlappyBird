import readchar

class KeyboardAction():
    def action(self):
        print "u for up";
        c = readchar.readchar()
        if c == 'u':
            return 1
        else:
            return 0
