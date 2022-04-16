(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(custom-enabled-themes (quote (tango-dark)))
 '(elpy-shell-echo-input nil)
 '(lean-rootdir "~/lean")
 '(org-latex-inputenc-alist (quote (("utf8" . "utf8x"))))
 '(org-latex-listings t)
 '(package-selected-packages
   (quote
    (let-alist sml-mode wolfram-mode org-ref tide org elpy magit company-lean company lean-mode)))
 '(python-indent-guess-indent-offset nil)
 '(python-indent-guess-indent-offset-verbose nil))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 )

;; General
(package-initialize)
(elpy-enable) ;; activating elpy automatically
;;(use-package org-tempo)
(toggle-scroll-bar -1)
(tool-bar-mode -1)
(global-linum-mode 1)

(defun settings ()
  (interactive)
  (find-file "~/.emacs"))

(require 'package) ; You might already have this line
(add-to-list 'package-archives '("melpa" . "http://melpa.org/packages/"))
(add-to-list 'package-archives '("org" . "http://orgmode.org/elpa/") t) ; Org-mode's repository

(package-initialize) ; You might already have this line

;; Trigger completion on Shift-Space
(global-set-key (kbd "S-SPC") #'company-complete)

(set-language-environment "UTF-8")
(set-default-coding-systems 'utf-8)
;; General

;; Lean 

(defun lean-kbd ()
  (local-set-key (kbd "C-c C-c") `comment-region)
  (local-set-key (kbd "C-c C-v") `uncomment-region)
  )

(add-hook `lean-mode-hook `lean-kbd)

;; Lean

;; magit

(global-set-key (kbd "C-x g") `magit-status)

;; magit

;; ocaml
(load "/home/minchao/.opam/system/share/emacs/site-lisp/tuareg-site-file")

;; ocaml

;; Latex

;; org-ref and bib see  https://github.com/jkitchin/org-ref/issues/165
;; solutions from John Kitchin@CMU
;; https://github.com/jkitchin/jmax/blob/master/jmax-org.el#L461
;; try removing -shell-escape if accent symbols like {'e} do not work
(require 'org-tempo)

(require 'org-ref)
(setq org-latex-pdf-process
      '("pdflatex -shell-escape -interaction nonstopmode -output-directory %o %b"
        "bibtex %b"
        "makeindex %b"
        "pdflatex -shell-escape -interaction nonstopmode -output-directory %o %b"
        "pdflatex -shell-escape -interaction nonstopmode -output-directory %o %b"))

;; Latex

;; HOL

(load "~/HOL/tools/hol-mode")

(load "~/proj/hol/hol-input.el")

(defun my-sml-mode-hook ()
  "Local defaults for SML mode"
  (setq electric-indent-chars '()))

(defun hol-kbd ()
  (local-set-key (kbd "C-c C-c") `comment-region)
  (local-set-key (kbd "C-c C-v") `uncomment-region))

(add-hook 'sml-mode-hook 'my-sml-mode-hook)
(add-hook 'sml-mode-hook 'hol-kbd)

(add-hook 'sml-mode-hook
	  (lambda () (set-input-method "Hol")))
(add-hook 'org-mode-hook
	  (lambda () (set-input-method "Hol")))


;;
(put 'upcase-region 'disabled nil)
