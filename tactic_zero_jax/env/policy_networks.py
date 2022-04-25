#!/usr/bin/env python
# coding: utf-8

# In[456]:


import haiku as hk
import jax
import optax
from jax import random
from jax import numpy as jnp
from jax import jit
from functools import partial


# In[457]:


#define networks for agent as in torch implementation

#should be given multiple encoded contexts (goals, assumptions) at once, so shape (batch, encoding_dim)

class ContextPolicy(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)    
        
        #can make deeper as with Minchao's later models
        self.fc = hk.Linear(512)
        self.fc2 = hk.Linear(1024)
        self.head = hk.Linear(1)
        #no sigmoid?
        
    def __call__(self, x):
        x = jax.nn.relu(self.fc(x)) 
        x = jax.nn.relu(self.fc2(x)) 
        
        x = self.head(x)

        return x

def _context_forward(x):
    module = ContextPolicy()
    return module(x)


# from old code, takes in just a single context encoding. Should it also take in the state?

class TacPolicy(hk.Module):
    def __init__(self, tactic_size, name=None):
        super().__init__(name=name)    
    
        self.fc = hk.Linear(512)
        self.fc2 = hk.Linear(1024)

        self.head = hk.Linear(tactic_size)
        
    def __call__(self, x):

        x = jax.nn.relu(self.fc(x)) 
        x = jax.nn.relu(self.fc2(x)) 
        x = jax.nn.softmax(self.head(x), axis=1) 
        
        return x

def _tac_forward(x, action_size):
    module = TacPolicy(action_size)
    return module(x)

#called one sample at a time 


class ArgPolicy(hk.Module):
    
    #tactic_size is integer for number of possible tactics given by tactic_pool, embedding dim will match size of AE so they can stack
    def __init__(self, tactic_size, embedding_dim, name=None):
        super().__init__(name=name)
        self.tactic_embeddings = hk.Embed(tactic_size, embedding_dim)
    
        self.lstm = hk.LSTM(embedding_dim)
        self.fc = hk.Linear(512)
        self.fc2 = hk.Linear(1024)
        self.head = hk.Linear(1)

    # x is the previously predicted argument / tactic.
    # candidates is a matrix of possible arguments concatenated with the hidden states.

    def __call__(self, x, candidates, hidden):
        
        #asserting x is integer  
        #if x.shape == torch.Size([1]):
            # x is a tactic
            # print("good")
        x = jnp.expand_dims(self.tactic_embeddings(x), 0)
        #else:
        #    x = x
        
        #x = jnp.reshape(x, (1,-1))
        
        s = jax.nn.relu(self.fc(candidates)) 
        s = jax.nn.relu(self.fc2(s))
        scores = self.head(s)
        
        o, hidden = self.lstm(x, hidden)
        
        return hidden, scores

def _arg_forward(x, candidates, hidden, tactic_size, embedding_dim):#, init_state=None):
    module = ArgPolicy(tactic_size, embedding_dim)
    return module(x, candidates, hidden)


class TermPolicy(hk.Module):
    def __init__(self, tactic_size, embedding_dim, name=None):
        super().__init__(name=name)  
        
        self.tactic_embeddings = hk.Embed(tactic_size, embedding_dim)
        
        self.fc = hk.Linear(512)
        self.fc2 = hk.Linear(1024)
        self.head = hk.Linear(1)

    def __call__(self, candidates, tac):
        #tac is just an index here, the embedding layer simplifies the one-hot 
        tac = self.tactic_embeddings(tac)#.view(1,-1)
        
        #concatenate the tactic to every candidate 
    
        tac_tensor = jnp.stack([tac for _ in range(candidates.shape[0])], 1)
        x  = jnp.concatenate([jnp.reshape(tac_tensor, candidates.shape), candidates], axis=1)
    
        x = jax.nn.relu(self.fc(x))
        x = jax.nn.relu(self.fc2(x))
        x = self.head(x)
        
        return x
    
    
    
def _term_forward(candidates, tac, tactic_size, embedding_dim):
    module = TermPolicy(tactic_size, embedding_dim)
    return module(candidates, tac)


# In[435]:



# init_context, apply_context = hk.transform(_context_forward)
# init_tac, apply_tac = hk.transform(_tac_forward)
# init_arg, apply_arg = hk.transform(_arg_forward)
# init_term, apply_term = hk.transform(_term_forward)


# In[436]:


# rng_key = random.PRNGKey(1009)

# batch_size = 10
# MAX_LEN = 256


# c_term = random.normal(rng_key, (6, MAX_LEN))

# x_tac = random.normal(rng_key, (1, MAX_LEN))

# #candidate network, shapes given from old repo. Matrix of encoded arguments

# c_arg = random.normal(rng_key, (8, MAX_LEN))

# #hidden dim size, will be set initialised as the target goal g 
# h0 = random.normal(rng_key, (1,MAX_LEN))
# #initial state, will be initialised as chosen tactic t 
# c0 = random.normal(rng_key, (1,MAX_LEN))

# init_state = hk.LSTMState(h0, c0)

# #arg takes int up to tac_size
# x_arg = 5

# x_context = random.normal(rng_key, (batch_size, 256))


# In[437]:


# TAC_SIZE = 10
# context_params = init_context(rng_key, x_context)
# tactic_params = init_tac(rng_key, x_tac, TAC_SIZE)
# arg_params = init_arg(rng_key, x_arg, c_arg, init_state, TAC_SIZE, MAX_LEN)
# term_params = init_term(rng_key, c_term, x_arg, TAC_SIZE, MAX_LEN)


# In[438]:


# apply_context = jax.jit(apply_context)
# apply_tac = partial(jax.jit, static_argnums=3)(apply_tac)
# apply_arg = partial(jax.jit, static_argnums=(5,6))(apply_arg)
# apply_term = partial(jax.jit, static_argnums=(4,5))(apply_term)


# out_context = apply_context(context_params, rng_key, x_context)
# out_tac = apply_tac(tactic_params, rng_key, x_tac, TAC_SIZE)
# out_arg = apply_arg(arg_params,  rng_key, x_arg, c_arg, init_state, TAC_SIZE, MAX_LEN)
# out_term = apply_term(term_params, rng_key,c_term, x_arg, TAC_SIZE, MAX_LEN)


# In[439]:


# print (out_context.shape,out_tac.shape, out_arg[1].shape, out_term.shape)


# In[453]:



# #can multiply learning rate by reward for each update to give the policy gradient after grad of log probs is done 
# context_lr = 1e-2
# tactic_lr = 1e-2 
# arg_lr = 1e-2
# term_lr = 1e-2

# context_optimiser = optax.rmsprop(context_lr)
# tactic_optimiser = optax.rmsprop(tactic_lr)
# arg_optimiser = optax.rmsprop(arg_lr)
# term_optimiser = optax.rmsprop(term_lr)

# opt_state_context = context_optimiser.init(context_params)
# opt_state_tactic = tactic_optimiser.init(tactic_params)
# opt_state_arg = arg_optimiser.init(arg_params)
# opt_state_term = term_optimiser.init(term_params)




    


# In[454]:


#gradient example. May need to construct separate function for each net during training 

# def compute_probs(params, net, *args):
#     probs = jax.nn.softmax(jnp.ravel(net(params,*args)))
#     logits = jnp.log(probs)
#     ind = random.categorical(rng_key, logits)
#     log_prob = logits[ind]
#     return log_prob

# grad = jax.grad(compute_probs)(term_params, apply_term, rng_key, c_term, x_arg,TAC_SIZE, MAX_LEN)

# updates, opt_state_term = term_optimiser.update(grad, opt_state_term)
# term_params = optax.apply_updates(term_params, updates)


# In[455]:


# grads_test = grad

# print (grads_test
#       )


# In[ ]:





# In[ ]:




