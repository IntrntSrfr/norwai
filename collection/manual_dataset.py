import pygame
import os
import time

pygame.init()

w = 1000
h = 700

files = os.listdir("./data/images")
screen = pygame.display.set_mode((w, h))

def image(i):
    img = pygame.image.load("./data/images/{}".format(files[i])).convert()
    screen.blit(img, (250,50))

#find index of 00627_1.jpg
i = 13926 - 1
i_new = 13926 - 1

#Start real world timer
start = time.time()
end = None
image(i)
pygame.display.flip()

avg = []

status = True
while (status):
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            status = False
            print("Current image: {}".format(files[i]))
            print("New start index is: {}".format(i_new))


        #Display the average time


        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_SPACE or e.key == 8:
                end = time.time()
                avg.append(end - start)
                print("Average time: {}".format(sum(avg)/len(avg)))
                start = time.time()
            if e.key == pygame.K_SPACE:
                i += 1
                i_new += 1
                image(i)
                pygame.display.flip()  
            if e.key == 8:
                os.remove("./data/images/{}".format(files[i]))
                i += 1
                image(i)
                pygame.display.flip()
 
# deactivates the pygame library
pygame.quit()
