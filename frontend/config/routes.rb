# config/routes.rb
Rails.application.routes.draw do
  root 'game#start'
  post 'game/create', to: 'game#create'
  get 'game', to: 'game#show'
  post 'game/answer', to: 'game#answer'

  # --- ADD THESE TWO NEW ROUTES ---
  get 'game/guess', to: 'game#guess' # Page to show the final guess
  post 'game/end', to: 'game#end_game' # Action to handle the outcome
end