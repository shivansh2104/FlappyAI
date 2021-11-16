import pygame
import neat
import time
import os
import random
import visualise

import Bird 
import Pipe
import Terrain

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 800

pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans",50)

def draw_window(win, birds, pipes, base, score):
    win.blit(Terrain.BG_IMG, (0,0))

    for pipe in pipes :
        pipe.draw(win)

    for bird in birds :
        bird.draw(win)
    
    text = STAT_FONT.render("Score: "+str(score),1,(255,255,255))
    win.blit(text, (WINDOW_WIDTH-10-text.get_width(),10))

    base.draw(win)
    pygame.display.update()

def main(genomes, config):
    
    nets = []
    ge = []
    birds = []
    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        birds.append(Bird.Bird(230,350))
        g.fitness = 0
        ge.append(g)

    base = Terrain.Base(730)
    pipes = [Pipe.Pipe(700)]
    win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    score = 0

    run = True
    while run :
        clock.tick(30)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        # bird move
        pipe_ind = 0
        if  len(birds)>0:
            if len(pipes)>1 and birds[0].x>pipes[0].x+pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.3

            output = nets[x].activate((bird.y,abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
            if output[0] >0.5 :
                bird.jump()
        
        add_pipe = False
        rem = []
        for pipe in pipes:
            for x,bird in enumerate(birds) :
                if pipe.collide(bird):
                    ge[x].fitness -= 0.5
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x<bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width()<0:
                rem.append(pipe)
            pipe.move()
        if add_pipe :
            score += 1
            for g in ge :
                g.fitness += 5
            pipes.append(Pipe.Pipe(600))

        for r in rem :
            pipes.remove(r)
        
        for x,bird in enumerate(birds) :
            if bird.y + bird.img.get_height() >= 730 or bird.y<0 :
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        base.move()
        draw_window(win, birds, pipes, base, score)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, \
            neat.DefaultReproduction, \
            neat.DefaultSpeciesSet, \
            neat.DefaultStagnation, \
            config_path )
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,50)
    visualise.draw_net(config, winner, True)
    visualise.plot_stats(stats, ylog=False, view=True)
    visualise.plot_species(stats, view=True)

if __name__=='__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
