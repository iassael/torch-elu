--[[

   Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)

   Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter

   Implemented by John-Alexander M. Assael (www.johnassael.com), 2015

]]--

local ELU, parent = torch.class('nn.ELU','nn.Module')

function ELU:__init(nOutputPlane)
   -- if no argument provided, use shared model (weight is scalar)
   self.nOutputPlane = nOutputPlane or 0
   self.weight = torch.Tensor(nOutputPlane or 1):fill(1)
   self.gradWeight = torch.Tensor(nOutputPlane or 1)
   
   self.gradWeightBatch = torch.Tensor()
   self.mask_lt = torch.ByteTensor()
   self.mask_ge = torch.ByteTensor()
   self.buffer = torch.Tensor()

   self.nInput = 0
   self.nBatch = 0

   parent.__init(self)
end

function ELU:reset()
   self.weight:fill(1)
end

function ELU:updateOutput(input)
   -- f = If[x < 0, a*(Exp[x] - 1), x]

   if input:dim() == 1 then
      self.nInput = input:nElement()
   else
      self.nBatch = input:size(1)
      self.nInput = input:size(2)
   end

   self.output:resizeAs(input):copy(input)

   self.mask_lt:resize(input:size()):copy(input:lt(0))
   self.mask_ge:resize(input:size()):copy(self.mask_lt:ne(1))

   if self.nOutputPlane == 0 then
      self.output[self.mask_lt] = self.output:maskedSelect(self.mask_lt):exp():add(-1):mul(self.weight[1])
   else
      if self.nBatch > 0 then
         self.buffer = self.weight:repeatTensor(self.nBatch, 1)
      else
         self.buffer = self.weight
      end
      self.output[self.mask_lt] = self.output:maskedSelect(self.mask_lt):exp():add(-1):cmul(self.buffer[self.mask_lt])
   end

   return self.output
end

function ELU:updateGradInput(input, gradOutput)
   -- If[x < 0, a Exp[x], 1]

   self.gradInput:resizeAs(input):copy(input)

   if self.nOutputPlane == 0 then
      self.gradInput[self.mask_lt] = self.gradInput:maskedSelect(self.mask_lt):exp():mul(self.weight[1])
   else
      self.gradInput[self.mask_lt] = self.gradInput:maskedSelect(self.mask_lt):exp():cmul(self.buffer[self.mask_lt])
   end
   self.gradInput[self.mask_ge] = 1

   self.gradInput:cmul(gradOutput)

   return self.gradInput
end

function ELU:accGradParameters(input, gradOutput, scale)
   -- If[x < 0, Exp[x] - 1, 0]

   scale = scale or 1.0

   self.gradWeightBatch:resizeAs(input):copy(input)
   self.gradWeightBatch[self.mask_lt] = self.gradWeightBatch:maskedSelect(self.mask_lt):exp():add(-1)
   self.gradWeightBatch[self.mask_ge] = 0

   self.gradWeightBatch:cmul(gradOutput)   

   if self.nOutputPlane == 0 then
      self.gradWeight:add(scale*self.gradWeightBatch:sum())
   else
      if self.nBatch > 0 then
         self.gradWeight:add(scale, self.gradWeightBatch:sum(1))
      else
         self.gradWeight:add(scale, self.gradWeightBatch)
      end
   end

   return self.gradWeight
end