(define (problem BLOCKS-7-0)
(:domain BLOCKS)
(:objects A B C D E F G - block)
(:INIT (CLEAR E) (CLEAR G) (ONTABLE A) (ONTABLE F) (ON E D) (ON D C) (ON C B) (ON B A) (ON G F) (HANDEMPTY))
(:goal (AND (ON A E) (ON E B) (ON B F) (ON F G) (ON G C) (ON C D)))
)