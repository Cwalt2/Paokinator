# app/controllers/game_controller.rb

class GameController < ApplicationController
  # Define the location of your Python API
  API_BASE_URL = 'http://127.0.0.1:5000'

  # Renders the start page
  def start
  end

  # Action to create a new game session
  def create
    response = HTTParty.post("#{API_BASE_URL}/game/start")
    if response.success?
      session[:game_id] = response.parsed_response['game_id']
      session[:question_count] = 0
      redirect_to game_path # Redirect to the main game page
    else
      flash[:error] = "Could not start a new game. Is the API server running?"
      redirect_to root_path
    end
  end

  # Action to show the current question
  def show
    game_id = session[:game_id]
    unless game_id
      redirect_to root_path and return
    end

    response = HTTParty.get("#{API_BASE_URL}/game/#{game_id}/question")
    @question_data = response.parsed_response
    @predictions = session[:predictions]
  end

  # Action to submit an answer and decide next step
  def answer
    game_id = session[:game_id]
    session[:question_count] += 1
    payload = { feature: params[:feature], answer: params[:answer] }.to_json

    response = HTTParty.post(
      "#{API_BASE_URL}/game/#{game_id}/answer",
      body: payload,
      headers: { 'Content-Type' => 'application/json' }
    )

    if response.success?
      predictions = response.parsed_response['predictions']
      session[:predictions] = predictions
      
      top_guess, top_prob = predictions.first
      
      if (top_prob > 0.75 && session[:question_count] >= 5) || session[:question_count] >= 15
        session[:final_guess] = top_guess
        redirect_to game_guess_path
      else
        redirect_to game_path
      end
    else
      flash[:error] = "There was an error submitting your answer."
      redirect_to game_path
    end
  end

  # Action to show the final guess
  def guess
    @final_guess = session[:final_guess]
    @predictions = session[:predictions]
  end

  # Action to end the game and learn if necessary
  def end_game
    game_id = session[:game_id]

    if params[:correct] == 'no' && params[:correct_animal].present?
      animal_name = params[:correct_animal]
      HTTParty.post(
        "#{API_BASE_URL}/game/#{game_id}/learn",
        body: { correct_animal: animal_name }.to_json,
        headers: { 'Content-Type' => 'application/json' }
      )
      flash[:notice] = "Thank you for teaching me about #{animal_name}!"
    else
      flash[:notice] = "I win! Great game!"
    end

    HTTParty.delete("#{API_BASE_URL}/game/#{game_id}/end")
    reset_session
    redirect_to root_path
  end
end