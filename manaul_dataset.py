import pygame;
import os;

pygame.init()

w = 1000
h = 700

screen = pygame.display.set_mode((w, h))

def image(i):
    img = pygame.image.load("./data/images/{:04d}.jpg".format(i)).convert()
    screen.blit(img, (250,50))

i = 127
image(i)
pygame.display.flip()

status = True
while (status):
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            status = False

        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_SPACE:
                i += 1
                image(i)
                pygame.display.flip()  
            if e.key == 8:
                os.remove("./data/images/{:04d}.jpg".format(i))
                i += 1
                image(i)
                pygame.display.flip()
 
# deactivates the pygame library
pygame.quit()

